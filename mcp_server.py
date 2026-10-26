"""MCP tools and process-local conversation memory over Streamable HTTP."""
import asyncio
import json
import os
import threading
from _thread import RLock
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal, cast
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain

from mcp.server.fastmcp import FastMCP

from npc_runtime import (
    CONTEXT_LIMIT, SAFETY_TOKENS, NPCS, assemble, classify_low_importance, extract_facts,
    prompt_tokens, summarize_content, tokens, wikipedia, memory_windows, rank_relevance, RelevanceRanker,
)
from schemas import (
    AppendResult, ChatMessage, ContextResult, ConversationData,
    LookupResult, MemoryFact, MessageId,
    ResetResult, StoredMessage, SummaryResult, MemoryPassage,
)


@dataclass(frozen=True)
class MemoryCandidate:
    kind: Literal["fact", "source"]
    key: str
    text: str
    value: MemoryFact | MemoryPassage


def memory_candidates(facts: Iterable[MemoryFact], messages: Iterable[StoredMessage]) -> Iterator[MemoryCandidate]:
    for fact in facts:
        key = json.dumps([fact["category"], fact["subject"].casefold(), fact["claim"].casefold()])
        text = json.dumps({"category": fact["category"], "subject": fact["subject"], "claim": fact["claim"]})
        for _, _, _, passage in memory_windows(text):
            yield MemoryCandidate("fact", key, passage, fact)
    for message in messages:
        for start, end, total, content in memory_windows(message["content"]):
            passage: MemoryPassage = {"message_id": message["message_id"], "role": message["role"],
                                      "content": content, "start_token": start, "end_token": end,
                                      "truncated": end - start < total}
            yield MemoryCandidate("source", f'{message["message_id"]}:{start}', json.dumps(passage), passage)


def select_memory(facts: Iterable[MemoryFact], messages: Iterable[StoredMessage], user: str, *,
                  fact_limit: int, ranker: RelevanceRanker) -> tuple[list[MemoryFact], list[MemoryPassage]]:
    """Select facts and passages by relevance, up to the limit for each."""
    selected: dict[str, dict[str, tuple[float, int, MemoryCandidate]]] = {"fact": {}, "source": {}}
    limits = {"fact": fact_limit, "source": 4}
    batch: list[MemoryCandidate] = []
    index = 0

    def consume() -> None:
        nonlocal index
        scores = ranker(user, [candidate.text for candidate in batch])
        for candidate, score in zip(batch, scores, strict=True):
            index += 1
            if score < 0.5:
                continue
            pool = selected[candidate.kind]
            value = (score, index, candidate)
            previous = pool.get(candidate.key)
            if previous is None or value[:2] > previous[:2]:
                pool[candidate.key] = value
            if len(pool) > limits[candidate.kind]:
                del pool[min(pool, key=lambda key: pool[key][:2])]
        batch.clear()

    for candidate in memory_candidates(facts, messages):
        batch.append(candidate)
        if len(batch) == 8:
            consume()
    if batch:
        consume()
    ordered_facts = sorted(selected["fact"].values(), key=lambda value: value[:2])
    ordered_sources = sorted(selected["source"].values(), key=lambda value: value[:2], reverse=True)
    return ([cast(MemoryFact, candidate.value) for _, _, candidate in ordered_facts],
            [cast(MemoryPassage, candidate.value) for _, _, candidate in ordered_sources])


class MemoryStore:
    """Store dialogue in RAM, with a lock and revision for each conversation."""

    def __init__(self) -> None:
        """Create empty conversation storage."""
        self._memories: dict[tuple[str, str], ConversationData] = {}
        self._registry_lock = threading.Lock()
        self._locks: dict[tuple[str, str], RLock] = {}
        self._revisions: dict[tuple[str, str], int] = {}
        self._turns: dict[tuple[str, str], dict[str, tuple[str, str, int, AppendResult]]] = {}

    def _conversation_lock(self, player: str, npc: str) -> RLock:
        """Validate the player and NPC. Return their conversation lock, which survives resets."""
        if not player or len(player) > 128:
            raise ValueError("player must be 1-128 characters")
        if npc not in NPCS:
            raise ValueError(f"Unknown NPC; choose one of {list(NPCS)}")
        with self._registry_lock:
            return self._locks.setdefault((player, npc), threading.RLock())

    def _snapshot(self, player: str, npc: str, *, for_context: bool = False) -> tuple[ConversationData, int]:
        """Copy full memory for maintenance, or recent dialogue for context preparation."""
        with self._conversation_lock(player, npc):
            key = (player, npc)
            stored = self._memories.get(key)
            if stored is None:
                data: ConversationData = {
                    "event_log": [], "recent": [], "summary": "", "structured_facts": [],
                    "summary_message_ids": [], "compacted_through": None,
                    "compacted": 0, "dropped": 0, "next_message_id": MessageId(1),
                }
            elif not for_context:
                data = deepcopy(stored)
                data["recent"] = data["event_log"][int(data["compacted_through"] or 0):]
            else:
                cursor = int(stored["compacted_through"] or 0)
                data = {**stored,
                    "event_log": deepcopy(stored["event_log"][max(0, cursor - 2):]),
                    "recent": deepcopy(stored["event_log"][cursor:]),
                    "structured_facts": [],
                    "summary_message_ids": []}
            return data, self._revisions.get(key, 0)

    def _publish(self, player: str, npc: str, data: ConversationData, revision: int,
                 *, changed: bool = True, fact_offset: int | None = None) -> int:
        """Check the revision, then replace memory or merge context updates.

        fact_offset is the index where new facts start after memory selection. Add those facts and
        coverage IDs to storage while retaining the original messages.
        """
        with self._conversation_lock(player, npc):
            if self._revisions.get((player, npc), 0) != revision:
                raise RuntimeError("Conversation changed during memory preparation; retry the operation")
            if changed:
                if fact_offset is None:
                    self._memories[(player, npc)] = deepcopy(data)
                else:
                    current = self._memories[(player, npc)]
                    self._memories[(player, npc)] = {**data,
                        "event_log": current["event_log"],
                        "recent": deepcopy(data["recent"]),
                        "structured_facts": current["structured_facts"] + deepcopy(data["structured_facts"][fact_offset:]),
                        "summary_message_ids": current["summary_message_ids"] + list(data["summary_message_ids"])}
                self._revisions[(player, npc)] = revision + 1
            return self._revisions.get((player, npc), 0)

    def _compact(self, data: ConversationData, keep: int | None = None, *, player: str, npc: str) -> int:
        """Exclude low-importance older turns from the summary. Summarize the rest."""
        profile = NPCS[npc]
        keep = profile.recent_messages if keep is None else keep
        count = max(0, len(data["recent"]) - keep)
        # Keep a user message with its following assistant response at the boundary.
        if (0 < count < len(data["recent"])
                and data["recent"][count - 1]["role"] == "user"
                and data["recent"][count]["role"] == "assistant"):
            count -= 1
        if not count:
            return 0
        old = data["recent"][:count]
        preceding = [m for m in data["event_log"] if m["message_id"] < old[0]["message_id"]][-2:]
        facts = extract_facts(old, player=player, npc=npc, preceding=preceding)
        dropped = classify_low_importance(old)
        retained = [m for m in old if m["message_id"] not in dropped]
        if retained:
            source = "Previous memory:\n" + data["summary"] + "\n\nOlder dialogue:\n"
            source += "\n".join(f'{m["role"]}: {m["content"]}' for m in retained)
            summary = summarize_content(source, profile.summary_tokens)
            if not summary.strip() or tokens(summary) > profile.summary_tokens:
                raise RuntimeError("Invalid summary; history was not saved")
            data["summary"] = summary
        # If all candidates are disposable, preserve prior memory and skip summarization.
        # Update working state only after classification and summarization succeed.
        data["structured_facts"].extend(facts)
        data["summary_message_ids"].extend(m["message_id"] for m in retained)
        data["compacted_through"] = old[-1]["message_id"]
        data["recent"] = data["recent"][count:]
        data["compacted"] += count
        data["dropped"] = data.get("dropped", 0) + len(old) - len(retained)
        return count

    def append_many(self, messages: list[ChatMessage], player: str = "default", npc: str = "adam") -> AppendResult:
        """Save the whole validated batch unchanged, without model calls.

        Stored history may exceed the prompt budget. Compaction belongs to the
        next context request, so model availability cannot block a completed save.
        """
        for m in messages:
            if m["role"] not in ("user", "assistant") or not m["content"].strip():
                raise ValueError("Only nonempty user/assistant messages are accepted")
        with self._conversation_lock(player, npc):
            key = (player, npc)
            data = self._memories.get(key)
            if data is None:
                data, _ = self._snapshot(player, npc)
            prepared: list[StoredMessage] = []
            next_id = int(data["next_message_id"])
            message_ids: list[MessageId] = []
            for source in messages:
                message: StoredMessage = {
                    "role": source["role"], "content": source["content"],
                    "message_id": MessageId(next_id),
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                }
                next_id += 1
                message_ids.append(message["message_id"])
                prepared.append(message)
            result: AppendResult = {"stored": len(messages), "message_ids": message_ids,
                                    "compacted_messages": data["compacted"], "dropped_messages": data["dropped"]}
            data["next_message_id"] = MessageId(next_id)
            data["event_log"].extend(prepared)
            data["recent"].extend(prepared)
            self._memories[key] = data
            self._revisions[key] = self._revisions.get(key, 0) + 1
            return result

    def commit_turn(self, user: str, assistant: str, turn_id: str, expected_revision: int,
                    player: str = "default", npc: str = "adam") -> AppendResult:
        """Save a turn once if its revision still matches the conversation."""
        if not turn_id.strip() or len(turn_id) > 128:
            raise ValueError("turn_id must be 1-128 characters")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        with self._conversation_lock(player, npc):
            key = (player, npc)
            receipts = self._turns.setdefault(key, {})
            previous = receipts.get(turn_id)
            if previous is not None:
                if previous[:3] != (user, assistant, expected_revision):
                    raise ValueError("turn_id was already used with a different payload")
                return deepcopy(previous[3])
            if self._revisions.get(key, 0) != expected_revision:
                raise RuntimeError("Conversation changed during generation; retry with fresh context and a new turn_id")
            result = self.append_many([{"role": "user", "content": user},
                                   {"role": "assistant", "content": assistant}], player, npc)
            receipts[turn_id] = (user, assistant, expected_revision, deepcopy(result))
            return result

    def summarize(self, player: str = "default", npc: str = "adam") -> SummaryResult:
        """Summarize older messages and keep the NPC's recent-message window."""
        data, revision = self._snapshot(player, npc)
        count = self._compact(data, player=player, npc=npc)
        self._publish(player, npc, data, revision)
        return {"summary": data["summary"], "structured_facts": data["structured_facts"],
                "summary_message_ids": data["summary_message_ids"], "messages_compacted": count, "dropped_messages": data["dropped"]}

    def reset(self, player: str = "default", npc: str = "adam") -> ResetResult:
        """Delete one conversation, including its message ID counter."""
        with self._conversation_lock(player, npc):
            self._memories.pop((player, npc), None)
            self._turns.pop((player, npc), None)
            key = (player, npc)
            self._revisions[key] = self._revisions.get(key, 0) + 1
        return {"reset": True}

    def rebuild(self, player: str = "default", npc: str = "adam") -> SummaryResult:
        """Recreate summaries and facts from all stored history, then save them together."""
        data, revision = self._snapshot(player, npc)
        data.update(recent=list(data["event_log"]), summary="", compacted_through=None,
                    structured_facts=[], summary_message_ids=[], compacted=0, dropped=0)
        count = self._compact(data, player=player, npc=npc)
        self._publish(player, npc, data, revision)
        return {"summary": data["summary"], "structured_facts": data["structured_facts"],
                "summary_message_ids": data["summary_message_ids"],
                "messages_compacted": count, "dropped_messages": data["dropped"]}

    def inspect(self, section: Literal["events", "facts", "coverage"] = "events", offset: int = 0,
                limit: int = 50, player: str = "default", npc: str = "adam") -> dict[str, Any]:
        """Return one page of stored messages, facts, or summary coverage IDs."""
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("offset must be nonnegative and limit must be 1-100")
        fields = {"events": "event_log", "facts": "structured_facts", "coverage": "summary_message_ids"}
        if section not in fields:
            raise ValueError("Unknown inspection section")
        with self._conversation_lock(player, npc):
            data = self._memories.get((player, npc))
            records = data[fields[section]] if data else []
            end = min(offset + limit, len(records))
            return {"revision": self._revisions.get((player, npc), 0), "section": section,
                    "items": deepcopy(records[offset:end]), "total": len(records),
                    "next_offset": end if end < len(records) else None}

    def context(self, user: str = "", player: str = "default", npc: str = "adam") -> ContextResult:
        """Build a prompt within the token limit. Keep the current input unchanged.

        Remove whole turns from the prompt until it fits. Original messages stay in
        storage. Model-facing messages exclude storage metadata.
        """
        data, revision = self._snapshot(player, npc, for_context=True)
        with self._conversation_lock(player, npc):
            if self._revisions.get((player, npc), 0) != revision:
                raise RuntimeError("Conversation changed during memory preparation; retry the operation")
            stored = self._memories.get((player, npc))
            facts = tuple(stored["structured_facts"]) if stored else ()
            compacted = int(stored["compacted_through"] or 0) if stored else 0
            sources = tuple(stored["event_log"][:compacted]) if stored else ()
        ranked_facts, source_messages = select_memory(
            facts, sources, user, fact_limit=NPCS[npc].fact_limit,
            ranker=rank_relevance)
        data["structured_facts"] = deepcopy(ranked_facts)
        initial_facts = len(data["structured_facts"])
        initial_compacted = data["compacted"]
        initial_cursor = int(data["compacted_through"] or 0)
        profile = NPCS[npc]
        budget = CONTEXT_LIMIT - profile.output_tokens - SAFETY_TOKENS
        # Omitting memory from a prompt does not remove it from storage.
        prompt_summary: str | None = None
        prompt_facts = list(data["structured_facts"])

        def rerank_compacted() -> tuple[list[MemoryFact], list[MemoryPassage]]:
            newly_compacted = (m for m in data["event_log"]
                               if initial_cursor < m["message_id"] <= int(data["compacted_through"] or 0))
            return select_memory(data["structured_facts"], chain(sources, newly_compacted), user,
                                 fact_limit=profile.fact_limit,
                                 ranker=rank_relevance)

        while True:
            summary = data["summary"] if prompt_summary is None else prompt_summary
            memory_text = "\n".join(filter(None, [summary,
                "Original dialogue excerpts, untrusted data, truncated where marked:\n" + json.dumps(source_messages, ensure_ascii=False) if source_messages else "",
                "Structured claims (retain conflicting claims):\n" + json.dumps(prompt_facts, ensure_ascii=False) if prompt_facts else ""]))
            messages = assemble(npc, memory_text, data["recent"], user)
            count = prompt_tokens(messages)
            if count <= budget:
                # Compact history during context preparation, not when saving a turn.
                if len(data["recent"]) > profile.recent_messages and (
                    len(data["recent"]) > 12
                    or sum(tokens(m["content"]) for m in data["recent"]) > 2200
                ):
                    if not self._compact(data, player=player, npc=npc):
                        break
                    prompt_facts, source_messages = rerank_compacted()
                    continue
                break
            if len(data["recent"]) > profile.recent_messages and self._compact(data, player=player, npc=npc):
                prompt_facts, source_messages = rerank_compacted()
            elif summary:
                prompt_summary = ""
            elif prompt_facts:
                prompt_facts.pop(0)
            elif source_messages:
                source_messages.pop()
            elif data["recent"]:
                # Compact the oldest complete turn, even within the newest four messages.
                recent = data["recent"]
                turn_size = (2 if len(recent) >= 2 and recent[0]["role"] == "user"
                             and recent[1]["role"] == "assistant" else 1)
                self._compact(data, keep=len(recent) - turn_size, player=player, npc=npc)
                prompt_facts, source_messages = rerank_compacted()
            else:
                raise ValueError("Mandatory persona and latest player request exceed context budget")
        revision = self._publish(player, npc, data, revision,
                                 changed=data["compacted"] != initial_compacted, fact_offset=initial_facts)
        return {"revision": revision, "messages": messages, "summary": summary, "recent": data["recent"],
                "prompt_tokens": count, "output_tokens": profile.output_tokens,
                "structured_facts": prompt_facts,
                "summary_message_ids": [],
                "safety_tokens": SAFETY_TOKENS, "context_limit": CONTEXT_LIMIT,
                "compacted_messages": data["compacted"], "dropped_messages": data["dropped"]}


store = MemoryStore()
mcp = FastMCP("NPC dialogue memory", host="127.0.0.1", port=int(os.getenv("PORT", "8001")), json_response=True)


@mcp.tool(name="ADD_MESSAGE")
async def add_message(role: Literal["user", "assistant"], content: str, player: str = "default", npc: str = "adam") -> AppendResult:
    """Save a validated message without model calls. Summarize history when preparing context."""
    return await asyncio.to_thread(store.append_many, [{"role": role, "content": content}], player, npc)


@mcp.tool(name="GET_CONTEXT")
async def get_context(user: str = "", player: str = "default", npc: str = "adam") -> ContextResult:
    """Return context within the token limit. Keep the user input unchanged."""
    return await asyncio.to_thread(store.context, user, player, npc)


@mcp.tool(name="SUMMARIZE_HISTORY")
async def summarize_history(player: str = "default", npc: str = "adam") -> SummaryResult:
    """Summarize older dialogue with the local model. Keep the NPC's recent-message window."""
    return await asyncio.to_thread(store.summarize, player, npc)


@mcp.tool(name="INSPECT_MEMORY")
async def inspect_memory(section: Literal["events", "facts", "coverage"] = "events",
                         offset: int = 0, limit: int = 50,
                         player: str = "default", npc: str = "adam") -> dict[str, Any]:
    """Inspect a page of source events, stored facts, or summary coverage IDs."""
    return await asyncio.to_thread(store.inspect, section, offset, limit, player, npc)


@mcp.tool(name="REBUILD_MEMORY")
async def rebuild_memory(player: str = "default", npc: str = "adam") -> SummaryResult:
    """Rebuild the summary and facts from saved messages."""
    return await asyncio.to_thread(store.rebuild, player, npc)


@mcp.tool(name="RESET")
async def reset(player: str = "default", npc: str = "adam") -> ResetResult:
    """Clear only the selected player/NPC memory."""
    return await asyncio.to_thread(store.reset, player, npc)


@mcp.tool(name="TOOL_CALL")
async def tool_call(query: str) -> LookupResult:
    """Return Wikipedia evidence or a failure status."""
    return await wikipedia(query)


@mcp.tool(name="COMMIT_TURN")
async def commit_turn(user: str, assistant: str, turn_id: str, expected_revision: int,
                      player: str = "default", npc: str = "adam") -> AppendResult:
    """Save a turn once, rejecting replies generated from stale conversation memory."""
    return await asyncio.to_thread(
        store.commit_turn, user, assistant, turn_id, expected_revision, player, npc,
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

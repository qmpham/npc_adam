"""Shared types for chat prompts, stored memory, and MCP results."""
from typing import Annotated, Literal, NewType, NotRequired
from pydantic import ConfigDict, Field, StringConstraints, TypeAdapter, with_config
from typing_extensions import TypedDict


MessageId = NewType("MessageId", int)


@with_config(ConfigDict(strict=True, extra="ignore"))
class TokenizeResponse(TypedDict):
    """The server token count. Ignore other response fields."""

    count: Annotated[int, Field(ge=0)]


TOKENIZE_RESPONSE_ADAPTER = TypeAdapter(TokenizeResponse)


@with_config(ConfigDict(extra="forbid", strict=True))
class ChatMessage(TypedDict):
    """A chat API message without storage metadata."""

    role: Literal["system", "user", "assistant"]
    content: str


class StoredMessage(ChatMessage):
    """A message with an increasing conversation ID and a UTC timestamp."""

    message_id: MessageId
    created_at: str


class MemoryPassage(TypedDict):
    """A token window with its source message and position."""

    message_id: MessageId
    role: Literal["system", "user", "assistant"]
    content: str
    start_token: int
    end_token: int
    truncated: bool


@with_config(ConfigDict(strict=True, extra="forbid"))
class RelevanceProbabilities(TypedDict):
    relevant: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    irrelevant: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


@with_config(ConfigDict(strict=True, extra="ignore"))
class RelevanceAnswer(TypedDict):
    probabilities: RelevanceProbabilities


@with_config(ConfigDict(strict=True, extra="ignore"))
class RelevanceResponse(TypedDict):
    answers: dict[str, RelevanceAnswer]


RELEVANCE_ADAPTER = TypeAdapter(RelevanceResponse)


@with_config(ConfigDict(extra="forbid", strict=True))
class LookupArguments(TypedDict):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


@with_config(ConfigDict(extra="forbid", strict=True))
class NativeFunction(TypedDict):
    name: Literal["TOOL_CALL"]
    arguments: str


@with_config(ConfigDict(extra="forbid", strict=True))
class NativeToolCall(TypedDict):
    id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    type: Literal["function"]
    function: NativeFunction


PromptMessage = ChatMessage
LOOKUP_ARGUMENTS_ADAPTER = TypeAdapter(LookupArguments)
NATIVE_CALL_ADAPTER = TypeAdapter(NativeToolCall)


@with_config(ConfigDict(extra="forbid", strict=True))
class MemoryClaim(TypedDict):
    """The schema for claims the model extracts."""

    category: Literal["goal", "relationship", "inventory", "quest", "event", "identity", "promise"]
    subject: str
    claim: str


@with_config(ConfigDict(extra="forbid", strict=True))
class FactExtraction(TypedDict):
    facts: Annotated[list[MemoryClaim], Field(max_length=6)]


class MemoryFact(MemoryClaim):
    """A claim with source message IDs. Conflicting claims have separate records."""

    source_message_ids: list[MessageId]


@with_config(ConfigDict(extra="forbid", strict=True))
class ConversationData(TypedDict):
    """Memory stored in this process for one player and NPC, including recent messages."""

    event_log: list[StoredMessage]
    structured_facts: list[MemoryFact]
    summary_message_ids: list[MessageId]
    summary: str
    compacted_through: MessageId | None
    compacted: int
    dropped: int
    next_message_id: MessageId
    recent: list[StoredMessage]


@with_config(ConfigDict(extra="forbid", strict=True))
class LookupSuccess(TypedDict):
    status: Literal["ok"]
    title: str
    url: str
    extract: str


@with_config(ConfigDict(extra="forbid", strict=True))
class LookupFailure(TypedDict):
    status: Literal["error", "not_found"]
    error: str


LookupResult = Annotated[LookupSuccess | LookupFailure, Field(discriminator="status")]


@with_config(ConfigDict(extra="forbid", strict=True))
class ContextResult(TypedDict):
    """A prompt within the token limit, its memory, and token counts."""

    revision: int
    messages: list[PromptMessage]
    summary: str
    structured_facts: NotRequired[list[MemoryFact]]
    summary_message_ids: NotRequired[list[MessageId]]
    recent: list[StoredMessage]
    prompt_tokens: int
    output_tokens: int
    safety_tokens: int
    context_limit: int
    compacted_messages: int
    dropped_messages: int


class AppendResult(TypedDict):
    """Message IDs assigned after saving the whole batch."""

    stored: int
    message_ids: list[MessageId]
    compacted_messages: int
    dropped_messages: int


class SummaryResult(TypedDict):
    """Summary text and the number of messages compacted by this operation."""

    summary: str
    structured_facts: NotRequired[list[MemoryFact]]
    summary_message_ids: NotRequired[list[MessageId]]
    messages_compacted: int
    dropped_messages: int


class ResetResult(TypedDict):
    """Confirmation that the selected conversation was cleared."""

    reset: bool


@with_config(ConfigDict(extra="forbid"))
class LookupDecision(TypedDict):
    """A factual request with a Wikipedia query that resolves conversation references."""

    action: Literal["lookup"]
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


@with_config(ConfigDict(extra="forbid"))
class DialogueDecision(TypedDict):
    """A request the NPC can answer without an external lookup."""

    action: Literal["dialogue"]


RouteDecision = LookupDecision | DialogueDecision


@with_config(ConfigDict(extra="forbid", strict=True))
class CandidateDecision(TypedDict):
    """The page ID the ranker selected, or None if search returned no pages."""

    page_id: int | None


ROUTE_ADAPTER = TypeAdapter(RouteDecision)

CONTEXT_ADAPTER = TypeAdapter(ContextResult)
LOOKUP_ADAPTER = TypeAdapter(LookupResult)
FACT_EXTRACTION_ADAPTER = TypeAdapter(FactExtraction)

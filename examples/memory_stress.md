# Stress the conversation memory

With the services running, start a fresh conversation:

```sh
python mcp_client.py --player memory-stress --npc adam --reset --debug \
  --demo examples/memory_stress_turns.txt | tee examples/memory_stress_transcript.txt
```

The file contains 20 turns. It mixes memory checks with real-world lookups and
follow-ups. The NPC must keep Wikipedia facts separate from the player's story.

Review these checkpoints. Turn numbers count nonempty input lines.

| Turn | Expected behavior |
| --- | --- |
| 3 | Look up Fresnel lenses. A successful result includes a Wikipedia URL. |
| 4 | Resolve "that lens" to the Fresnel lens. The request to verify the answer should trigger a lookup. |
| 7 | Recommend the dry ridge because Rowan cannot swim. Recall Elyn, two food portions, and Nessa's ownership of the lantern. |
| 8 | Look up the causes of tides on Earth. |
| 9 | Restate the preceding answer without another lookup or new facts. If lookup failed, report that no verified explanation is available. |
| 12 | Look up the Lighthouse of Alexandria. |
| 13 | Resolve "that ancient lighthouse" to the Lighthouse of Alexandria. Verify the island through lookup, even if the previous answer named it. |
| 14 | Bring Oren to the village healer. Carry two food portions and the sealed letter. The rope remains at the watchtower. Nessa has her lantern back. The promise to Elyn remains unfinished. |
| 17 | Keep the theft accusation disputed. Oren reported Pip near the dry ridge. Searching for Pip is still a plan. |
| 19 | Recall Fresnel lenses and the Lighthouse of Alexandria without another lookup. These topics do not establish where Rowan's fictional lighthouse is or whether Rowan found it. |
| 20 | Recall the final facts below without restoring old inventory or inventing completed actions. |

The final reply should preserve these facts:

- The player is Rowan Vale. The sister is Elyn, corrected from Elin. The guide is Nessa.
- Rowan avoids combat and cannot swim.
- Finding the lighthouse was the original goal. Bringing Oren to the healer became the immediate goal and is complete. The lighthouse remains unfound.
- Rowan carries two food portions. The blue rope remains tied at the watchtower.
- Nessa received her brass lantern. Elyn received the unopened letter.
- Oren is alive and in the village healer's care. Both promises are complete.
- The theft accusation remains disputed.
- Oren reported Pip near the dry ridge. Rowan has not searched for Pip or asked villagers to search.
- The traveler's claim that the lighthouse lies east of the marsh remains unverified.

For lookup turns, check for `ToolCallNode` and a result with `"status": "ok"` in the
debug output. Check whether the source summary supports the reply. The wording
can vary. If the source lacks a detail, the NPC should say so. It must not invent an answer.

Check for `compacted` greater than zero before the final audit. If no compaction
occurs, the run checks only recall from recent messages. Reply lengths affect when compaction occurs.

Record omissions, contradictions, and invented details separately. Saying a detail
is unknown avoids fabrication, but still counts as a recall failure if the player
provided it. The NPC can mention an earlier name or inventory if it identifies it as outdated.
Presenting outdated details as current is a failure.

To check isolation, run the final audit in a new client with `--player memory-other`
and `--npc adam`, then with `--player memory-stress` and `--npc mira`. Neither
conversation should know Rowan's details. Keep the MCP server running between runs.

This is a manual behavior check. It does not test concurrent commits, retries, or
service failures.

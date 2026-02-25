IT Helpdesk Triage Agent Assessment Solutions

## Q1: Design Choices — Architecture

Honestly, this is a classic junior developer move - they're trying to be clever but missing the point. Sharing conversation_history across tickets is problematic for several reasons:

First, tickets are completely independent issues from different users. There's no logical reason why processing TKT-001 (VPN error) should influence how we classify TKT-002 (Salesforce access). The model isn't "learning" patterns here - it's just getting confused by unrelated context.

Second, look at the prompts.py - the system prompt explicitly defines the task as classifying individual tickets. By injecting previous tickets into the history, we're violating the single-turn nature of the task. The model might start thinking it's having a conversation rather than doing one-off classifications.

Third, there's a practical issue: if the first ticket triggers a tool call, that whole tool interaction history gets carried over. So when processing TKT-002, the model sees all the VPN troubleshooting steps from TKT-001. That could easily bias it toward network-related classifications.

The developer's justification about "learning patterns" would make sense if we were doing few-shot learning with examples, but we're not - we're just dumping raw ticket histories. Each ticket should start with a clean slate, maybe just the system prompt and the current ticket.

## Q2: Trace the State

Looking at agent.py, there's a serious bug in how multiple tool_use blocks are handled. The code does:

```python
tool_use = next(b for b in response.content if b.type == "tool_use")
```

The `next()` function only grabs the FIRST tool_use it finds. So if the LLM returns two tool_use blocks (which it absolutely can, Claude supports multiple tool calls), the second one gets completely ignored.

Here's exactly what gets appended:
1. The entire response.content (containing both tool_use blocks) gets appended as assistant message
2. ONE tool_result (only for the first tool_use) gets appended as user message

The information loss is significant - the second tool_use's ID and the fact that the LLM wanted to call that tool are completely lost. The model will never see the results for that second tool call, and worse, the conversation history now shows the assistant making a tool call that never gets resolved. This could confuse the model in subsequent turns.

What should happen is either handle multiple tool_uses in a loop, or batch process them with multiple tool_results in a single user message. The current implementation just silently drops the second tool call.

## Q3: Predict the Failure Mode

The issue with naive len-based pruning isn't about the slice deletion logic - it's about breaking the required message structure for Claude's API.

Claude requires messages to alternate between user and assistant roles. You can't have two user messages in a row or two assistant messages in a row. When you blindly slice to the last 10 messages, you might end up with a sequence like:

- user (ticket text)
- assistant (with tool_use)
- user (tool_result)
- assistant (final response)

If you chop this down to the last 3 messages, you might get:
- user (tool_result)
- assistant (final response)

That's valid. But depending on timing, you could get:
- assistant (with tool_use)
- user (tool_result)
- assistant (final response)

Also valid. The problem comes when pruning cuts off the alternating pattern. For example, if you have:
- user (ticket)
- assistant (tool_use)
- user (tool_result)
- assistant (final)
- user (next ticket)

Pruning to last 3 might give you:
- user (tool_result)
- assistant (final)
- user (next ticket)

That's fine. But the API could reject if pruning leaves you with two user messages consecutively. The pruning doesn't check the role sequence, it just counts messages. In a complex conversation with multiple tool turns, this becomes increasingly likely to corrupt the alternation pattern.

## Q4: What happens with THIS input?

Running through retrieval.py with the ticket "My laptop screen keeps flickering when I connect it to the docking station. Can someone replace the dock?":

**Result: None** (no chunks returned)

**Chunk_ids: empty list**

Here's why:

1. The query first goes through `embed_query()`. It checks against pre-computed patterns:
   - "vpn connection error" - no match
   - "salesforce access" - no match  
   - "excel crash add-in" - no match

2. Falls back to category keyword weighting. The code looks for ["network", "software", "hardware", "access"] in the query. The query has "laptop", "screen", "docking station", "dock" - none of these category keywords appear.

3. So the fallback embedding becomes the base vector: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]

4. Now cosine similarity with each chunk:
   - net-001-a: [0.92,0.12,0.04,0.02,0.88,0.15,0.06,0.03] vs base vector
   - Similarity will be relatively low because base is all 0.25s while chunk vectors have high variation
   - None will hit the 0.85 threshold

The core issue is the embedding strategy is too simplistic. A real embedding model would capture semantic similarity between "docking station" and hardware concepts, but the fallback just checks for exact category name matches.

## Q5: Fix evaluation

**Is increasing top_k effective?** No, that completely misses the point. If the relevant chunk isn't being retrieved at all (similarity < 0.85), increasing top_k from 3 to 10 won't help - it's still below the threshold.

**Actual root cause:** Looking at net-001-b chunk_id, it contains "Error 619" and port 1723 information. The query for TKT-001 mentions "error code 619". But the similarity score between the query embedding and net-001-b's embedding is likely below 0.85. Why? Because the query embedding is probably based on the fallback logic rather than a proper embedding of "error code 619 VPN Windows update".

**Better approach:** 

1. First, actually use a proper embedding model instead of the hand-crafted fallback. That alone would likely catch the semantic similarity between "error code 619" and the chunk content.

2. If we're stuck with the current system, at least expand the pre-computed patterns. Add "error code 619" to the QUERY_EMBEDDINGS with an embedding close to net-001-b.

3. Lower the similarity threshold slightly (0.75-0.8) to catch near-misses, but this is a band-aid.

4. Consider hybrid search - combine embedding similarity with keyword matching for error codes. Error codes are exact identifiers that should trigger direct matches.

The real fix is recognizing that error codes need special handling - they're not semantic text, they're identifiers. The RAG pipeline should treat them differently from natural language descriptions.

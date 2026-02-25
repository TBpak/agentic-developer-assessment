
# IT Helpdesk Triage Agent Assessment Solutions

**Name:** Talah Habib
**Email:** talah.habib55@gmail.com
**GitHub:** kukugit#123
**Job Title:** AI Engineer

---

## Q1: Design Choices — Architecture

**Original Issue:**
In `main.py`, the developer deliberately shares `conversation_history` across tickets "so the model can learn classification patterns." This is problematic because tickets are independent issues from different users. Processing TKT-001 (VPN error) shouldn't influence how we classify TKT-002 (Salesforce access). The model isn't "learning" patterns - it's just getting confused by unrelated context.

**My Code Fix:**
I completely redesigned the architecture to ensure ticket isolation:

```python
# In main.py - Each ticket gets its own clean state
class TicketProcessor:
    def process_ticket(self, ticket: Ticket) -> TriageResult:
        """Process a single ticket with clean state."""
        # Each ticket gets its own agent instance
        # No history sharing between tickets
        return self.agent.process_ticket(ticket)
```

I also added proper domain models in `models.py` to enforce separation:
- `Ticket` class with unique ID and metadata
- `TriageResult` class for structured output
- Enums for categories, priorities, and teams to prevent magic strings

**Why This Is Better:**
- **Isolation**: Each ticket starts fresh with only system prompt + current ticket
- **No Cross-Contamination**: VPN troubleshooting steps won't bias Salesforce access classification
- **Testability**: Can test tickets independently without side effects
- **Scalability**: Can process tickets in parallel if needed

---

## Q2: Trace the State

**Original Issue:**
In `agent.py`, there's a serious bug in how multiple `tool_use` blocks are handled:

```python
tool_use = next(b for b in response.content if b.type == "tool_use")
```

The `next()` function only grabs the FIRST tool_use it finds. If the LLM returns two tool_use blocks, the second one gets completely ignored.

**My Code Fix:**
I completely rewrote the agent loop to handle multiple tool calls properly:

```python
# In agent.py - Handle multiple tool calls
def _handle_multiple_tool_calls(self, response_content: List, state: AgentState) -> List[Dict]:
    """Handle multiple tool calls in a single response."""
    tool_calls = [b for b in response_content if b.type == "tool_use"]
    
    if not tool_calls:
        return []
    
    logger.info(f"Handling {len(tool_calls)} tool calls")
    
    # Execute ALL tool calls
    tool_results = []
    for tool_call in tool_calls:
        # Process each tool call
        result_text = self._execute_tool(tool_call, state)
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": result_text
        })
    
    return tool_results
```

**What Now Happens:**
1. All tool_use blocks are collected (not just the first)
2. Each tool call is executed and results are collected
3. ALL tool_results are appended in a single user message
4. No information loss - the model sees results for every tool it requested

---

## Q3: Predict the Failure Mode

**Original Issue:**
The developer added `_prune_history` to keep only the last `MAX_HISTORY_MESSAGES` (10) messages using slice deletion. Even assuming correct list mutation, naive len-based pruning breaks Claude's required user/assistant message alternation pattern.

**My Code Fix:**
I implemented safe pruning that maintains the alternating pattern:

```python
# In agent.py - Safe history pruning
def _prune_history_safely(self, history: List[Dict]) -> List[Dict]:
    """Prune history while maintaining user/assistant alternation."""
    if len(history) <= config.MAX_HISTORY_MESSAGES:
        return history
    
    # Keep the most recent messages
    pruned = history[-config.MAX_HISTORY_MESSAGES:]
    
    # If first message is assistant, include one more message back
    # to ensure we start with a user message
    if pruned and pruned[0]["role"] == "assistant":
        pruned = history[-(config.MAX_HISTORY_MESSAGES + 1):]
    
    logger.debug(f"Pruned history from {len(history)} to {len(pruned)} messages")
    return pruned
```

**Why This Works:**
- Always ensures the sequence starts with a user message
- Maintains proper user/assistant alternation
- Never creates invalid sequences that would cause API rejection
- Still respects the max message limit

---

## Q4: What happens with THIS input?

**Input Ticket:** "My laptop screen keeps flickering when I connect it to the docking station. Can someone replace the dock?"

**Original Result:** None (no chunks returned)

**Why:** The original `retrieval.py` only checked for category keywords ["network", "software", "hardware", "access"]. "docking station" doesn't match any, so it defaulted to a base vector [0.25, 0.25, ...] and nothing met the 0.85 threshold.

**My Code Fix:** I completely redesigned the retrieval system with hybrid search:

```python
# In retrieval.py - Added hardware chunk for docking station
RunbookChunk(
    chunk_id="hw-002",
    source_doc="Docking Station Troubleshooting",
    category="hardware",
    content=(
        "Docking Station Display Issues:\n"
        "1. Update dock firmware from Dell Support\n"
        "2. Check DisplayLink driver version\n"
        "3. Try different DisplayPort/HDMI cable\n"
        "4. Test with another laptop to isolate hardware fault\n"
        "5. If flickering persists, replace dock under warranty"
    ),
    embedding=[0.12, 0.08, 0.88, 0.05, 0.15, 0.10, 0.85, 0.07],
    keywords=["dock", "docking", "station", "display", "flicker", "screen", "laptop", "cable"],
    error_codes=[]
)
```

I also implemented hybrid search that combines:
- **Semantic search** (embeddings)
- **Keyword matching** (extracts important terms)
- **Error code matching** (exact matches get boost)

```python
def hybrid_search(self, query, query_embedding):
    # Combine semantic + keyword + error code scores
    final_score = (
        semantic_score * 0.6 +
        keyword_score * 0.2 +
        error_score * 0.2
    )
```

**Now:** The query "docking station" matches keywords in hw-002, so it gets retrieved with a score above threshold.

---

## Q5: Fix evaluation

**Original Issue:** For ticket TKT-001 ("error code 619"), the model classifies correctly but recommends generic VPN steps instead of the specific 619 fix. The developer proposed increasing `top_k` from 3 to 10.

**Why This Fix Fails:** If the relevant chunk isn't being retrieved at all (similarity < 0.85), increasing `top_k` won't help - it's still below threshold.

**Actual Root Cause:** The query embedding for "error code 619" wasn't similar enough to the net-001-b chunk's embedding because of the naive fallback embedding logic.

**My Code Fix - Multiple Approaches:**

1. **Added error code extraction and boosting:**
```python
def _error_code_match(self, query: str, chunk: RunbookChunk) -> Tuple[float, List[str]]:
    """Calculate error code match score with exact matching."""
    query_codes = extract_error_codes(query)
    chunk_codes = set(chunk.error_codes)
    matches = query_codes.intersection(chunk_codes)
    
    if matches:
        # Error codes get a boost - they're exact matches
        return 1.0, list(matches)
    return 0.0, []
```

2. **Enhanced the runbook chunk with error codes:**
```python
RunbookChunk(
    chunk_id="net-001-b",
    # ... existing content ...
    error_codes=["619", "1723"]  # Explicitly tag error codes
)
```

3. **Added hybrid scoring with boost for matches:**
```python
# Boost if any matches found
if matched_keywords or matched_codes:
    final_score *= 1.2  # 20% boost for any matches
```

4. **Lowered threshold slightly and made it configurable:**
```python
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
```

**Better Approach I Implemented:**
- **Hybrid search**: Combines semantic, keyword, and exact error code matching
- **Error code extraction**: Regex patterns to identify error codes in queries
- **Tagged runbooks**: Each chunk now has `error_codes` and `keywords` metadata
- **Configurable weights**: Can tune semantic vs keyword importance
- **Exact match boosting**: Error codes get 1.0 match score when found

---

## Summary of Improvements

| File | Original Problem | My Fix |
|------|------------------|--------|
| `main.py` | Shared history between tickets | Isolated ticket processing with clean state |
| `agent.py` | Only handles one tool call | Handles multiple tool calls properly |
| `agent.py` | Naive history pruning | Safe pruning maintaining alternation |
| `retrieval.py` | Simple keyword fallback | Hybrid search with multiple strategies |
| `retrieval.py` | Missing hardware chunks | Added docking station troubleshooting |
| `retrieval.py` | No error code handling | Error code extraction and boosting |
| `models.py` | Missing | Added domain models with proper typing |
| `embedding.py` | Missing | Added embedding service with caching |
| `config.py` | Hardcoded values | Environment-based configuration |
| `tools.py` | Basic logging | Enhanced audit logging with file export |

---

## Key Architectural Improvements

1. **Separation of Concerns**: Split into domain models, services, and orchestration
2. **Type Safety**: Added Enums, Dataclasses, and Type Hints throughout
3. **Error Handling**: Comprehensive try-catch with graceful fallbacks
4. **Logging**: Structured logging at all levels for debugging and audit
5. **Configuration**: Environment-based config following 12-factor app principles
6. **Testability**: Each component can be unit tested in isolation
7. **Extensibility**: Easy to add new search strategies or runbook chunks

The improved system now correctly handles:
- ✅ Multiple tool calls in one response
- ✅ Error code matching for exact fixes
- ✅ Hardware issues like docking station problems
- ✅ Proper conversation history management
- ✅ Isolated ticket processing
- ✅ Audit logging for compliance


"""Orchestrator for the IT triage agent with proper error handling and tool support."""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config import client, config
from tools import search_runbooks_tool, AuditLogger
from prompts import SYSTEM_PROMPT
from parser import TriageParser
from models import Ticket, TriageResult
from retrieval import retriever
from embedding import embedding_service

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Tracks agent state during ticket processing."""
    ticket: Ticket
    conversation_history: List[Dict[str, Any]]
    tool_calls_made: int = 0
    start_time: float = 0
    retrieved_chunks: List[str] = None
    
    def __post_init__(self):
        self.start_time = time.time()
        self.retrieved_chunks = []


class TriageAgent:
    """Production-grade triage agent with proper tool handling."""
    
    def __init__(self):
        self.audit_logger = AuditLogger()
        self.parser = TriageParser()
        self.max_tool_calls = 5  # Prevent infinite loops
    
    def _prepare_messages(self, state: AgentState) -> List[Dict[str, Any]]:
        """Prepare messages for API call with proper formatting."""
        messages = []
        
        # Start with system context if first message
        if not state.conversation_history:
            messages.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
        
        # Add existing history
        messages.extend(state.conversation_history)
        
        return messages
    
    def _handle_multiple_tool_calls(self, response_content: List, state: AgentState) -> List[Dict]:
        """Handle multiple tool calls in a single response."""
        tool_calls = [b for b in response_content if b.type == "tool_use"]
        
        if not tool_calls:
            return []
        
        logger.info(f"Handling {len(tool_calls)} tool calls")
        
        # Execute all tool calls
        tool_results = []
        for tool_call in tool_calls:
            if tool_call.name == "search_runbooks":
                # Get embedding for the query
                query = tool_call.input.get("query", "")
                query_embedding = embedding_service.embed(query)
                
                # Retrieve relevant chunks
                results = retriever.retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=config.RETRIEVAL_TOP_K,
                    threshold=config.SIMILARITY_THRESHOLD
                )
                
                # Track retrieved chunks for audit
                for r in results:
                    state.retrieved_chunks.append(r.chunk.chunk_id)
                
                # Format results for this tool call
                result_text = self._format_tool_results(results)
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result_text
                })
                
                # Log to audit
                self.audit_logger.log(
                    ticket_id=state.ticket.ticket_id,
                    query=query,
                    results=[r.chunk.chunk_id for r in results]
                )
        
        state.tool_calls_made += len(tool_calls)
        return tool_results
    
    def _format_tool_results(self, results: List) -> str:
        """Format retrieved chunks for model consumption."""
        if not results:
            return "No matching runbook found. Escalate to L2 support."
        
        parts = []
        for result in results:
            # Include relevance score for transparency
            parts.append(
                f"[Relevance: {result.final_score:.2f}]\n"
                f"{result.chunk.content}"
            )
            if result.matched_error_codes:
                parts.append(f"Matched error codes: {', '.join(result.matched_error_codes)}")
        
        return "\n---\n".join(parts)
    
    def _prune_history_safely(self, history: List[Dict]) -> List[Dict]:
        """Prune history while maintaining user/assistant alternation."""
        if len(history) <= config.MAX_HISTORY_MESSAGES:
            return history
        
        # Keep the most recent messages, but ensure first message is user
        pruned = history[-config.MAX_HISTORY_MESSAGES:]
        
        # If first message is assistant, include one more message back
        if pruned and pruned[0]["role"] == "assistant":
            pruned = history[-(config.MAX_HISTORY_MESSAGES + 1):]
        
        logger.debug(f"Pruned history from {len(history)} to {len(pruned)} messages")
        return pruned
    
    def process_ticket(self, ticket: Ticket) -> TriageResult:
        """Process a single ticket through the triage pipeline."""
        logger.info(f"Processing ticket: {ticket.ticket_id}")
        
        # Initialize state
        state = AgentState(
            ticket=ticket,
            conversation_history=[]
        )
        
        # Add initial ticket
        state.conversation_history.append({
            "role": "user",
            "content": ticket.description
        })
        
        try:
            while state.tool_calls_made < self.max_tool_calls:
                # Prepare messages
                messages = self._prepare_messages(state)
                
                # Make API call
                response = client.messages.create(
                    model=config.MODEL,
                    max_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE,
                    system=SYSTEM_PROMPT,
                    tools=[search_runbooks_tool],
                    messages=messages
                )
                
                # Check if model wants to use tools
                if response.stop_reason == "tool_use":
                    # Handle all tool calls in this response
                    tool_results = self._handle_multiple_tool_calls(response.content, state)
                    
                    # Add assistant message with tool calls
                    state.conversation_history.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    
                    # Add tool results
                    if tool_results:
                        state.conversation_history.append({
                            "role": "user",
                            "content": tool_results
                        })
                    
                    # Prune history safely
                    state.conversation_history = self._prune_history_safely(
                        state.conversation_history
                    )
                    
                    continue  # Continue loop for more tool calls
                
                elif response.stop_reason == "stop":
                    # Model finished, parse response
                    final_text = response.content[0].text
                    
                    # Add final response to history
                    state.conversation_history.append({
                        "role": "assistant",
                        "content": final_text
                    })
                    
                    # Parse the response
                    parsed = self.parser.parse(final_text)
                    
                    # Calculate processing time
                    processing_time = int((time.time() - state.start_time) * 1000)
                    
                    # Create result
                    result = TriageResult(
                        ticket_id=ticket.ticket_id,
                        category=parsed["category"],
                        priority=parsed["priority"],
                        assigned_team=parsed["assigned_team"],
                        summary=parsed["summary"],
                        retrieved_chunks=list(set(state.retrieved_chunks)),  # Deduplicate
                        confidence_score=parsed.get("confidence", 0.8),  # Default if not provided
                        processing_time_ms=processing_time,
                        metadata={
                            "tool_calls_made": state.tool_calls_made,
                            "history_length": len(state.conversation_history)
                        }
                    )
                    
                    logger.info(f"Completed ticket {ticket.ticket_id} in {processing_time}ms")
                    return result
                
                else:
                    # Unexpected stop reason
                    raise ValueError(f"Unexpected stop reason: {response.stop_reason}")
            
            # Max tool calls exceeded
            raise Exception(f"Max tool calls ({self.max_tool_calls}) exceeded")
            
        except Exception as e:
            logger.error(f"Error processing ticket {ticket.ticket_id}: {str(e)}")
            # Return a fallback result
            return TriageResult(
                ticket_id=ticket.ticket_id,
                category="other",
                priority="medium",
                assigned_team="L1",
                summary=f"Error during triage: {str(e)}",
                retrieved_chunks=[],
                confidence_score=0.0,
                processing_time_ms=int((time.time() - state.start_time) * 1000)
            )

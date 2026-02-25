"""Tool definitions with audit logging and monitoring."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AuditLogger:
    """Audit logging for compliance and monitoring."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.audit_log = []
    
    def log(self, ticket_id: str, query: str, results: List[str]):
        """Log a tool call for audit purposes."""
        entry = {
            "ticket_id": ticket_id,
            "query": query,
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
            "environment": "production"  # Track environment
        }
        
        self.audit_log.append(entry)
        
        # Also log to file if configured
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Audit log: Ticket {ticket_id} searched for '{query[:50]}...'")
    
    def get_logs(self, ticket_id: Optional[str] = None) -> List[Dict]:
        """Retrieve audit logs, optionally filtered by ticket ID."""
        if ticket_id:
            return [log for log in self.audit_log if log["ticket_id"] == ticket_id]
        return list(self.audit_log)
    
    def export_to_json(self, filepath: str):
        """Export audit logs to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.audit_log, f, indent=2)


# Tool definition following Anthropic's tool schema
search_runbooks_tool = {
    "name": "search_runbooks",
    "description": (
        "Search the internal IT runbook database for troubleshooting "
        "steps relevant to a support ticket. Use this tool when you need "
        "to find known solutions or troubleshooting procedures."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticket_id": {
                "type": "string",
                "description": "The unique ID of the support ticket for audit tracking",
            },
            "query": {
                "type": "string",
                "description": "Detailed search query describing the technical issue. Include error codes, symptoms, and affected systems.",
                "minLength": 5,
                "maxLength": 500
            },
            "category_hint": {
                "type": "string",
                "enum": ["network", "software", "hardware", "access", "unknown"],
                "description": "Optional category hint to narrow the search",
                "default": "unknown"
            }
        },
        "required": ["ticket_id", "query"],
    }
}

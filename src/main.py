"""Main entry point for batch IT helpdesk ticket triage.

Usage:
    python main.py
    python main.py --tickets tickets.json --output results.json
"""

import argparse
import json
import logging
import sys
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from agent import TriageAgent
from models import Ticket, TriageResult
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'triage_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)


class TicketProcessor:
    """Process tickets in batch with proper isolation."""
    
    def __init__(self, agent: TriageAgent):
        self.agent = agent
        self.results: List[TriageResult] = []
    
    def process_ticket(self, ticket: Ticket) -> TriageResult:
        """Process a single ticket with clean state."""
        # Each ticket gets its own agent instance
        # No history sharing between tickets
        return self.agent.process_ticket(ticket)
    
    def process_batch(self, tickets: List[Ticket]) -> List[TriageResult]:
        """Process multiple tickets in batch."""
        logger.info(f"Starting batch processing of {len(tickets)} tickets")
        
        for i, ticket in enumerate(tickets, 1):
            try:
                logger.info(f"Processing ticket {i}/{len(tickets)}: {ticket.ticket_id}")
                result = self.process_ticket(ticket)
                self.results.append(result)
                
                logger.info(f"Completed {ticket.ticket_id}: "
                           f"{result.category.value} - {result.priority.value}")
                
            except Exception as e:
                logger.error(f"Failed to process {ticket.ticket_id}: {str(e)}")
                # Continue with next ticket
                continue
        
        logger.info(f"Batch processing complete. Processed {len(self.results)}/{len(tickets)} tickets")
        return self.results
    
    def export_results(self, filepath: str):
        """Export results to JSON file."""
        output = []
        for r in self.results:
            output.append({
                "ticket_id": r.ticket_id,
                "category": r.category.value,
                "priority": r.priority.value,
                "assigned_team": r.assigned_team.value,
                "summary": r.summary,
                "retrieved_chunks": r.retrieved_chunks,
                "confidence_score": r.confidence_score,
                "processing_time_ms": r.processing_time_ms,
                "metadata": r.metadata
            })
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results exported to {filepath}")


def load_tickets_from_file(filepath: str) -> List[Ticket]:
    """Load tickets from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    tickets = []
    for item in data:
        ticket = Ticket.create(
            description=item["description"],
            **item.get("metadata", {})
        )
        tickets.append(ticket)
    
    return tickets


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="IT Helpdesk Triage Agent")
    parser.add_argument(
        "--tickets",
        type=str,
        help="JSON file containing tickets to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="triage_results.json",
        help="Output file for results (default: triage_results.json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create agent
    agent = TriageAgent()
    processor = TicketProcessor(agent)
    
    if args.tickets:
        # Load tickets from file
        tickets = load_tickets_from_file(args.tickets)
    else:
        # Use default test tickets
        tickets = [
            Ticket.create(
                "My laptop won't connect to the VPN. I've tried restarting multiple times "
                "and keep getting error code 619. This started after the Windows update last night."
            ),
            Ticket.create(
                "I need access to the Salesforce sandbox environment for UAT testing next week. "
                "My manager has approved this — ref: approval email from Sarah dated 12 Feb."
            ),
            Ticket.create(
                "Excel keeps crashing whenever I open files larger than 10MB. "
                "Running Office 365, Windows 11. Started happening after I added a new add-in."
            ),
            # Add the docking station ticket
            Ticket.create(
                "My laptop screen keeps flickering when I connect it to the docking station. Can someone replace the dock?"
            )
        ]
    
    # Process tickets
    results = processor.process_batch(tickets)
    
    # Export results
    processor.export_results(args.output)
    
    # Print summary
    print(f"\nProcessed {len(results)} tickets successfully")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

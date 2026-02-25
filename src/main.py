"""Main entry point for batch IT helpdesk ticket triage.

Run with:
```
python main.py
```
"""

import utils.session
import json
from agent import triage_ticket

def main():

    conversation_history = []
    
    tickets = [
        (
            "TKT-001",
            "My laptop won't connect to the VPN. I've tried restarting multiple times "
            "and keep getting error code 619. This started after the Windows update last night."
        ),
        (
            "TKT-002",
            "I need access to the Salesforce sandbox environment for UAT testing next week. "
            "My manager has approved this — ref: approval email from Sarah dated 12 Feb."
        ),
        (
            "TKT-003",
            "Excel keeps crashing whenever I open files larger than 10MB. "
            "Running Office 365, Windows 11. Started happening after I added a new add-in."
        ),
    ]

    for ticket_id, ticket_text in tickets:
        print(f"\nProcessing {ticket_id}...")
        
        # We share the history so the model can learn classification patterns 
        # across the sequence of tickets during this batch run.
        result = triage_ticket(ticket_text, ticket_id, conversation_history)
        
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

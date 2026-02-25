"""Domain models for the IT Helpdesk Triage Agent.
Separates data structures from business logic.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class TicketCategory(str, Enum):
    """Strictly defined ticket categories."""
    NETWORK = "network"
    SOFTWARE = "software"
    HARDWARE = "hardware"
    ACCESS = "access"
    OTHER = "other"


class TicketPriority(str, Enum):
    """Priority levels with clear definitions."""
    CRITICAL = "critical"  # System down, multiple users affected
    HIGH = "high"          # Individual blocker, no workaround
    MEDIUM = "medium"       # Individual issue with workaround
    LOW = "low"            # Nice-to-have, not blocking


class AssignedTeam(str, Enum):
    """Support teams with escalation paths."""
    L1 = "L1"              # Level 1 - Basic troubleshooting
    L2 = "L2"              # Level 2 - Specialized technical
    L3 = "L3"              # Level 3 - Engineering/Development
    SECURITY = "security"   # Security team
    NETWORK = "network"     # Network team
    HARDWARE = "hardware"   # Hardware team


@dataclass
class Ticket:
    """Represents a support ticket."""
    ticket_id: str
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, description: str, **metadata) -> "Ticket":
        """Factory method to create a new ticket with generated ID."""
        return cls(
            ticket_id=f"TKT-{uuid.uuid4().hex[:8].upper()}",
            description=description,
            metadata=metadata
        )


@dataclass
class TriageResult:
    """Structured output from the triage process."""
    ticket_id: str
    category: TicketCategory
    priority: TicketPriority
    assigned_team: AssignedTeam
    summary: str
    retrieved_chunks: List[str]  # Chunk IDs used for reference
    confidence_score: float  # 0-1 score for the classification
    processing_time_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunbookChunk:
    """A single chunk from a runbook document with embedding."""
    chunk_id: str
    source_doc: str
    category: str
    content: str
    embedding: List[float]
    keywords: List[str] = field(default_factory=list)  # For hybrid search
    error_codes: List[str] = field(default_factory=list)  # Specific error codes

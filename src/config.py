"""Configuration for the IT Helpdesk Triage Agent.
Follows 12-factor app principles with environment variable support.
"""

import os
from dataclasses import dataclass
from typing import Final, Optional
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass(frozen=True)
class AgentConfig:
    """Immutable configuration settings for the agent."""
    
    # API Configuration
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # Model settings
    MODEL: str = os.getenv("MODEL", "claude-3-sonnet-20240229")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))  # Keep deterministic for triage
    
    # Conversation management
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
    
    # RAG retrieval settings
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    
    # Embedding settings
    EMBEDDING_DIM: Final[int] = 8  # Matches current dummy embeddings
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        if not 0 <= self.TEMPERATURE <= 1:
            raise ValueError("TEMPERATURE must be between 0 and 1")
        if self.MAX_HISTORY_MESSAGES < 2:
            raise ValueError("MAX_HISTORY_MESSAGES must be at least 2 for alternating roles")

# Global configuration instance
config = AgentConfig()

# Initialize client once and reuse
client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

"""Advanced RAG retrieval pipeline with hybrid search and proper embeddings.

Uses a combination of:
- Semantic search (embeddings)
- Keyword matching for error codes and exact terms
- Configurable thresholds and boost factors
"""

import math
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import Counter

from models import RunbookChunk
from config import config

# Set up logging
logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Search strategy types."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    """Structured search result with metadata."""
    chunk: RunbookChunk
    semantic_score: float
    keyword_score: float
    final_score: float
    matched_keywords: List[str] = None
    matched_error_codes: List[str] = None


class RunbookRetriever:
    """Production-grade retriever with hybrid search capabilities."""
    
    def __init__(self, chunks: List[RunbookChunk]):
        """Initialize retriever with runbook chunks."""
        self.chunks = chunks
        self._build_indices()
        logger.info(f"Initialized retriever with {len(chunks)} chunks")
    
    def _build_indices(self):
        """Build search indices for fast retrieval."""
        # In production, this would use vector databases like Pinecone, Weaviate, or pgvector
        # For now, we maintain in-memory indices
        self.keyword_index = {}
        self.error_code_index = {}
        
        for chunk in self.chunks:
            # Index keywords
            for keyword in chunk.keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(chunk)
            
            # Index error codes
            for code in chunk.error_codes:
                if code not in self.error_code_index:
                    self.error_code_index[code] = []
                self.error_code_index[code].append(chunk)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from query."""
        # Simple keyword extraction - in production, use NLP libraries
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "my", "can", "when", "keep", "started"}
        words = text.lower().split()
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def _extract_error_codes(self, text: str) -> List[str]:
        """Extract error codes using regex patterns."""
        import re
        # Match common error code patterns (619, 0x80070005, etc.)
        patterns = [
            r'\b\d{3,4}\b',  # 3-4 digit codes (619, 404)
            r'0x[0-9A-Fa-f]+',  # Hex codes
            r'error[:\s]*(\w+)',  # error: XYZ
        ]
        
        codes = []
        for pattern in patterns:
            codes.extend(re.findall(pattern, text, re.IGNORECASE))
        return list(set(codes))
    
    def _keyword_match(self, query: str, chunk: RunbookChunk) -> Tuple[float, List[str]]:
        """Calculate keyword match score."""
        query_keywords = set(self._extract_keywords(query))
        if not query_keywords:
            return 0.0, []
        
        chunk_keywords = set(chunk.keywords)
        matches = query_keywords.intersection(chunk_keywords)
        
        if matches:
            score = len(matches) / len(query_keywords)
            return score, list(matches)
        return 0.0, []
    
    def _error_code_match(self, query: str, chunk: RunbookChunk) -> Tuple[float, List[str]]:
        """Calculate error code match score with exact matching."""
        query_codes = set(self._extract_error_codes(query))
        if not query_codes:
            return 0.0, []
        
        chunk_codes = set(chunk.error_codes)
        matches = query_codes.intersection(chunk_codes)
        
        if matches:
            # Error codes get a boost - they're exact matches
            return 1.0, list(matches)
        return 0.0, []
    
    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity with numerical stability."""
        a = np.array(vec_a, dtype=np.float64)
        b = np.array(vec_b, dtype=np.float64)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        # Clip to handle floating point errors
        similarity = np.dot(a, b) / (norm_a * norm_b)
        return float(np.clip(similarity, -1.0, 1.0))
    
    def semantic_search(self, query_embedding: List[float]) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        results = []
        
        for chunk in self.chunks:
            semantic_score = self.cosine_similarity(query_embedding, chunk.embedding)
            results.append(SearchResult(
                chunk=chunk,
                semantic_score=semantic_score,
                keyword_score=0.0,
                final_score=semantic_score
            ))
        
        return sorted(results, key=lambda x: x.semantic_score, reverse=True)
    
    def hybrid_search(
        self, 
        query: str, 
        query_embedding: List[float],
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.2,
        error_code_weight: float = 0.2
    ) -> List[SearchResult]:
        """Perform hybrid search combining multiple strategies."""
        results = []
        
        # Pre-extract query features
        query_keywords = self._extract_keywords(query)
        query_error_codes = self._extract_error_codes(query)
        
        logger.debug(f"Query keywords: {query_keywords}")
        logger.debug(f"Query error codes: {query_error_codes}")
        
        for chunk in self.chunks:
            # Semantic score
            semantic_score = self.cosine_similarity(query_embedding, chunk.embedding)
            
            # Keyword match score
            keyword_score, matched_keywords = self._keyword_match(query, chunk)
            
            # Error code match score (exact matches get boost)
            error_score, matched_codes = self._error_code_match(query, chunk)
            
            # Weighted combination
            final_score = (
                semantic_score * semantic_weight +
                keyword_score * keyword_weight +
                error_score * error_code_weight
            )
            
            # Boost if any matches found
            if matched_keywords or matched_codes:
                final_score *= 1.2  # 20% boost for any matches
            
            results.append(SearchResult(
                chunk=chunk,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                final_score=final_score,
                matched_keywords=matched_keywords,
                matched_error_codes=matched_codes
            ))
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results
    
    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 3,
        threshold: float = 0.6,
        strategy: SearchStrategy = SearchStrategy.HYBRID
    ) -> List[SearchResult]:
        """Main retrieval method with configurable strategy."""
        
        if strategy == SearchStrategy.SEMANTIC:
            results = self.semantic_search(query_embedding)
        else:
            results = self.hybrid_search(query, query_embedding)
        
        # Filter by threshold and take top_k
        filtered = [r for r in results if r.final_score >= threshold]
        
        logger.info(f"Retrieved {len(filtered)} chunks above threshold {threshold}")
        return filtered[:top_k]


# Initialize with enhanced runbook chunks
ENHANCED_RUNBOOK_CHUNKS = [
    RunbookChunk(
        chunk_id="net-001-a",
        source_doc="VPN Troubleshooting Guide",
        category="network",
        content=(
            "VPN Connection Issues — General Steps:\n"
            "1. Verify VPN client version is up to date\n"
            "2. Check firewall rules on the endpoint\n"
            "3. Restart network adapter via Device Manager"
        ),
        embedding=[0.92, 0.12, 0.04, 0.02, 0.88, 0.15, 0.06, 0.03],
        keywords=["vpn", "connection", "firewall", "network", "adapter"],
        error_codes=[]
    ),
    RunbookChunk(
        chunk_id="net-001-b",
        source_doc="VPN Troubleshooting Guide",
        category="network",
        content=(
            "VPN Connection Issues — Error Codes:\n"
            "4. Flush DNS: run ipconfig /flushdns\n"
            "5. Error 619: port 1723 is blocked — check router\n"
            "   settings and contact the network infrastructure team"
        ),
        embedding=[0.85, 0.10, 0.06, 0.03, 0.80, 0.12, 0.08, 0.05],
        keywords=["vpn", "dns", "error", "port", "router"],
        error_codes=["619", "1723"]
    ),
    # Add hardware chunk for docking station issues
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
    ),
    # Rest of chunks with enhanced metadata...
]

# Initialize global retriever
retriever = RunbookRetriever(ENHANCED_RUNBOOK_CHUNKS)

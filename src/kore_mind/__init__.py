"""kore-mind: Persistent memory + emergent identity engine for any LLM."""

from kore_mind.models import Memory, Identity, MemoryType
from kore_mind.mind import Mind

__version__ = "0.1.0"
__all__ = ["Mind", "Memory", "Identity", "MemoryType"]

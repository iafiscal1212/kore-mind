"""kore-mind: Persistent memory + emergent identity engine for any LLM."""

from kore_mind.models import Memory, Identity, MemoryType, Trace, CacheEntry
from kore_mind.mind import Mind
from kore_mind.embeddings import numpy_embed, ollama_embed, ollama_embed_async, openai_embed

__version__ = "0.4.0"
__all__ = [
    "Mind", "Memory", "Identity", "MemoryType", "Trace", "CacheEntry",
    "numpy_embed", "ollama_embed", "ollama_embed_async", "openai_embed",
]

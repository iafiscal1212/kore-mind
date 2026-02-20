"""Mind: the core class. 5 methods. That's it."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from kore_mind.consolidate import consolidate
from kore_mind.decay import DEFAULT_HALF_LIFE, apply_decay
from kore_mind.models import Identity, Memory, MemoryType
from kore_mind.storage import Storage

# Type: takes text, returns embedding as bytes (numpy float32 .tobytes())
EmbedFn = Callable[[str], bytes]


class Mind:
    """Una mente persistente. Un archivo SQLite = una mente.

    API:
        mind.experience(text)   — algo pasó, registrar
        mind.recall(query)      — qué es relevante ahora
        mind.reflect(fn)        — consolidar, decaer, evolucionar
        mind.identity()         — quién soy ahora
        mind.forget(threshold)  — poda explícita
    """

    def __init__(self, path: str | Path = "mind.db",
                 half_life: float = DEFAULT_HALF_LIFE,
                 embed_fn: EmbedFn | None = None) -> None:
        self._storage = Storage(path)
        self._half_life = half_life
        self._embed_fn = embed_fn

    # ── experience ─────────────────────────────────────────────────────

    def experience(self, content: str,
                   type: str | MemoryType = MemoryType.EPISODIC,
                   source: str = "",
                   tags: list[str] | None = None,
                   salience: float = 1.0) -> Memory:
        """Registra una experiencia. Algo pasó."""
        if isinstance(type, str):
            type = MemoryType(type)

        embedding = None
        if self._embed_fn is not None:
            embedding = self._embed_fn(content)

        mem = Memory(
            content=content,
            type=type,
            source=source,
            tags=tags or [],
            salience=salience,
            embedding=embedding,
        )
        self._storage.save_memory(mem)
        return mem

    # ── recall ─────────────────────────────────────────────────────────

    def recall(self, query: str = "", limit: int = 20,
              min_salience: float = 0.05) -> list[Memory]:
        """Recupera recuerdos relevantes.

        Sin query: devuelve los más salientes.
        Con query + embed_fn: búsqueda semántica por cosine similarity.
        Con query sin embed_fn: búsqueda por substring match.
        """
        memories = self._storage.top_memories(
            limit=limit * 3,
            min_salience=min_salience,
        )

        if query and self._embed_fn is not None:
            # Semantic search
            query_emb = self._embed_fn(query)
            from kore_mind.embeddings import semantic_search
            scored = semantic_search(query_emb, memories, limit=limit)
            # Combine semantic score with salience
            result = []
            for sim, mem in scored:
                combined = sim * 0.7 + mem.salience * 0.3
                result.append((combined, mem))
            result.sort(key=lambda x: x[0], reverse=True)
            result = [mem for _, mem in result[:limit]]
        elif query:
            # Text search fallback — only reinforce actual matches
            query_lower = query.lower()
            scored = []
            for mem in memories:
                relevance = 0.0
                content_lower = mem.content.lower()
                if query_lower in content_lower:
                    relevance += 1.0
                for tag in mem.tags:
                    if query_lower in tag.lower():
                        relevance += 0.5
                score = mem.salience + relevance
                scored.append((score, relevance, mem))

            scored.sort(key=lambda x: x[0], reverse=True)
            result = [(rel, mem) for _, rel, mem in scored[:limit]]
        else:
            result = [(1.0, mem) for mem in memories[:limit]]

        # Acceder refuerza — only if the memory actually matched
        final = []
        for relevance, mem in result:
            if relevance > 0:
                mem.access()
                self._storage.save_memory(mem)
            final.append(mem)

        return final

    # ── reflect ────────────────────────────────────────────────────────

    def reflect(self, summarizer: Callable[[list[Memory]], Identity] | None = None) -> Identity:
        """Consolida la mente: decay + consolidación + poda + identidad emergente.

        Args:
            summarizer: función opcional que recibe recuerdos y genera Identity.
                        Aquí es donde kore-bridge conecta un LLM.
                        Sin summarizer, genera identidad básica por estadísticas.
        """
        all_mems = self._storage.all_memories()

        # 1. Decay
        alive, dead = apply_decay(all_mems, self._half_life)

        # 2. Eliminar muertos
        for mem in dead:
            self._storage.delete_memory(mem.id)

        # 3. Consolidación: fusionar recuerdos similares
        use_emb = self._embed_fn is not None
        alive, deleted_ids = consolidate(
            alive, threshold=0.7, use_embeddings=use_emb,
        )
        for did in deleted_ids:
            self._storage.delete_memory(did)

        # 4. Guardar estado actualizado
        for mem in alive:
            self._storage.save_memory(mem)

        # 5. Generar identidad
        if summarizer is not None:
            identity = summarizer(alive)
        else:
            identity = self._default_identity(alive)

        identity.updated_at = time.time()
        self._storage.save_identity(identity)
        return identity

    # ── identity ───────────────────────────────────────────────────────

    def identity(self) -> Identity:
        """Quién soy ahora. Lee la última identidad generada por reflect()."""
        return self._storage.load_identity()

    # ── forget ─────────────────────────────────────────────────────────

    def forget(self, threshold: float = 0.1) -> int:
        """Olvida recuerdos con salience bajo el umbral. Devuelve cuántos."""
        return self._storage.delete_below_salience(threshold)

    # ── utilidades ─────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Cuántos recuerdos hay."""
        return self._storage.count()

    def close(self) -> None:
        self._storage.close()

    def __enter__(self) -> Mind:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Mind(memories={self.count})"

    # ── identidad por defecto (sin LLM) ───────────────────────────────

    @staticmethod
    def _default_identity(memories: list[Memory]) -> Identity:
        """Genera identidad básica por estadísticas, sin LLM."""
        if not memories:
            return Identity(summary="No memories yet.")

        type_counts: dict[str, int] = {}
        sources: dict[str, int] = {}
        all_tags: dict[str, int] = {}

        for mem in memories:
            type_counts[mem.type.value] = type_counts.get(mem.type.value, 0) + 1
            if mem.source:
                sources[mem.source] = sources.get(mem.source, 0) + 1
            for tag in mem.tags:
                all_tags[tag] = all_tags.get(tag, 0) + 1

        sorted_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
        traits = {tag: count / len(memories) for tag, count in sorted_tags[:10]}

        top_tags = [t[0] for t in sorted_tags[:5]]
        dominant_type = max(type_counts, key=type_counts.get)
        summary = (
            f"{len(memories)} memories ({dominant_type}-dominant). "
            f"Focus: {', '.join(top_tags) if top_tags else 'general'}."
        )

        relationships = {src: f"{count} interactions" for src, count in sources.items()}

        return Identity(
            traits=traits,
            summary=summary,
            relationships=relationships,
        )

"""Mind: the core class. 5 methods. That's it."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from kore_mind.decay import DEFAULT_HALF_LIFE, apply_decay
from kore_mind.models import Identity, Memory, MemoryType
from kore_mind.storage import Storage


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
                 half_life: float = DEFAULT_HALF_LIFE) -> None:
        self._storage = Storage(path)
        self._half_life = half_life

    # ── experience ─────────────────────────────────────────────────────

    def experience(self, content: str,
                   type: str | MemoryType = MemoryType.EPISODIC,
                   source: str = "",
                   tags: list[str] | None = None,
                   salience: float = 1.0) -> Memory:
        """Registra una experiencia. Algo pasó."""
        if isinstance(type, str):
            type = MemoryType(type)

        mem = Memory(
            content=content,
            type=type,
            source=source,
            tags=tags or [],
            salience=salience,
        )
        self._storage.save_memory(mem)
        return mem

    # ── recall ─────────────────────────────────────────────────────────

    def recall(self, query: str = "", limit: int = 20,
              min_salience: float = 0.05) -> list[Memory]:
        """Recupera recuerdos relevantes.

        Sin query: devuelve los más salientes.
        Con query: busca por contenido (substring match simple).

        Para búsqueda semántica con embeddings, usa kore-bridge.
        """
        memories = self._storage.top_memories(
            limit=limit * 3,  # fetch extra, filter down
            min_salience=min_salience,
        )

        if query:
            query_lower = query.lower()
            scored = []
            for mem in memories:
                # Score simple: substring match + tag match + salience
                score = mem.salience
                content_lower = mem.content.lower()
                if query_lower in content_lower:
                    score += 1.0
                for tag in mem.tags:
                    if query_lower in tag.lower():
                        score += 0.5
                scored.append((score, mem))

            scored.sort(key=lambda x: x[0], reverse=True)
            result = [mem for _, mem in scored[:limit]]
        else:
            result = memories[:limit]

        # Acceder refuerza
        for mem in result:
            mem.access()
            self._storage.save_memory(mem)

        return result

    # ── reflect ────────────────────────────────────────────────────────

    def reflect(self, summarizer: Callable[[list[Memory]], Identity] | None = None) -> Identity:
        """Consolida la mente: decay + poda + identidad emergente.

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

        # 3. Guardar decayed salience
        for mem in alive:
            self._storage.save_memory(mem)

        # 4. Generar identidad
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

        # Contar tipos
        type_counts: dict[str, int] = {}
        sources: dict[str, int] = {}
        all_tags: dict[str, int] = {}

        for mem in memories:
            type_counts[mem.type.value] = type_counts.get(mem.type.value, 0) + 1
            if mem.source:
                sources[mem.source] = sources.get(mem.source, 0) + 1
            for tag in mem.tags:
                all_tags[tag] = all_tags.get(tag, 0) + 1

        # Top tags como traits
        sorted_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
        traits = {tag: count / len(memories) for tag, count in sorted_tags[:10]}

        # Summary
        top_tags = [t[0] for t in sorted_tags[:5]]
        dominant_type = max(type_counts, key=type_counts.get)
        summary = (
            f"{len(memories)} memories ({dominant_type}-dominant). "
            f"Focus: {', '.join(top_tags) if top_tags else 'general'}."
        )

        # Relationships from sources
        relationships = {src: f"{count} interactions" for src, count in sources.items()}

        return Identity(
            traits=traits,
            summary=summary,
            relationships=relationships,
        )

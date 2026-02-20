"""Core data models. A Memory has a lifecycle. An Identity emerges."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class MemoryType(str, Enum):
    EPISODIC = "episodic"      # experiencias, conversaciones
    SEMANTIC = "semantic"       # hechos, conocimiento
    PROCEDURAL = "procedural"  # cómo hacer cosas


@dataclass
class Memory:
    """Un recuerdo con ciclo de vida. No es texto estático — decae y evoluciona."""

    content: str
    type: MemoryType = MemoryType.EPISODIC
    salience: float = 1.0       # 0.0-1.0, decae con el tiempo
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    source: str = ""            # quién/qué generó este recuerdo
    tags: list[str] = field(default_factory=list)
    embedding: bytes | None = None  # optional embedding vector (numpy .tobytes())
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def access(self) -> None:
        """Acceder a un recuerdo lo refuerza."""
        self.access_count += 1
        self.last_accessed = time.time()
        # Refuerzo: salience sube un poco, nunca pasa de 1.0
        self.salience = min(1.0, self.salience + 0.05)


@dataclass
class Identity:
    """Identidad emergente. No se configura — se calcula de los recuerdos."""

    traits: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    relationships: dict[str, str] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

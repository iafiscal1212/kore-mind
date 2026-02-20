"""Salience decay. Memories that aren't used fade away."""

from __future__ import annotations

import math
import time

from kore_mind.models import Memory

# Half-life en segundos: la salience se reduce a la mitad cada HALF_LIFE
DEFAULT_HALF_LIFE = 7 * 24 * 3600  # 7 días


def compute_decay(mem: Memory, now: float | None = None,
                  half_life: float = DEFAULT_HALF_LIFE) -> float:
    """Calcula la nueva salience tras decaimiento exponencial.

    Fórmula: salience * 2^(-elapsed / half_life)
    Recuerdos accedidos frecuentemente resisten más (bonus).
    """
    if now is None:
        now = time.time()

    elapsed = now - mem.last_accessed
    if elapsed <= 0:
        return mem.salience

    # Bonus por uso frecuente: cada acceso duplica el half_life efectivo
    effective_half_life = half_life * (1 + math.log1p(mem.access_count))

    decayed = mem.salience * math.pow(2, -elapsed / effective_half_life)
    return max(0.0, min(1.0, decayed))


def apply_decay(memories: list[Memory], half_life: float = DEFAULT_HALF_LIFE,
                death_threshold: float = 0.01) -> tuple[list[Memory], list[Memory]]:
    """Aplica decay a una lista de recuerdos.

    Returns:
        (alive, dead) - recuerdos vivos y recuerdos que cayeron bajo el umbral
    """
    now = time.time()
    alive = []
    dead = []

    for mem in memories:
        mem.salience = compute_decay(mem, now, half_life)
        if mem.salience < death_threshold:
            dead.append(mem)
        else:
            alive.append(mem)

    return alive, dead

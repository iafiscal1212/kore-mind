#!/usr/bin/env python3
"""
kore-mind demo: The full cognitive lifecycle.

No LLM needed. No API keys. Just run it.
"""

import os
import tempfile
import time

from kore_mind import Mind


def header(text):
    print(f"\n{'='*64}")
    print(f"  {text}")
    print(f"{'='*64}\n")


def show(mind, label=""):
    memories = mind._storage.all_memories(min_salience=0.0)
    memories.sort(key=lambda m: m.salience, reverse=True)
    if label:
        print(f"  [{label}] {len(memories)} memories:")
    for m in memories:
        n = int(m.salience * 20)
        bar = "█" * n + "░" * (20 - n)
        status = ""
        if m.salience < 0.05:
            status = " ← dying"
        print(f"    {bar} {m.salience:.2f} | {m.content[:55]}{status}")
    print()


def main():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    mind = Mind(db_path, half_life=1.5)  # 1.5s ≈ 1 week

    header("KORE-MIND: Cognitive Lifecycle Demo")
    print("  1.5 seconds ≈ 1 simulated week.\n")

    # ── Week 1 ─────────────────────────────────────────────────────────

    header("WEEK 1 — First interactions")

    mind.experience(
        "User researches computational complexity (P vs NP)",
        type="semantic", source="carlos", tags=["research", "complexity"],
    )
    mind.experience(
        "User prefers concise and direct answers",
        type="semantic", source="carlos", tags=["preference"],
    )
    mind.experience(
        "User uses Python for proof verification",
        type="semantic", source="carlos", tags=["python", "proofs"],
    )
    mind.experience(
        "User said good morning",
        type="episodic", source="carlos", tags=["greeting"],
    )
    mind.experience(
        "User asked about the weather",
        type="episodic", source="carlos", tags=["smalltalk"],
    )

    show(mind, "Week 1 — all fresh at 1.00")

    # ── Week 2 ─────────────────────────────────────────────────────────

    header("WEEK 2 — Reinforce important, ignore trivial")

    time.sleep(2)

    # Reinforce what matters
    mind.recall("complexity")
    mind.recall("python")

    # Repetition (should consolidate with earlier)
    mind.experience(
        "User prefers concise and direct responses",
        type="semantic", source="carlos", tags=["preference"],
    )
    mind.experience(
        "User building proof engine called AIP",
        type="semantic", source="carlos", tags=["AIP", "proofs"],
    )

    show(mind, "Week 2 — trivial memories decaying")

    # ── Week 3: Reflect ────────────────────────────────────────────────

    header("WEEK 3 — Reflect: decay + consolidation + identity")

    time.sleep(2)

    before = mind.count
    identity = mind.reflect()
    after = mind.count

    delta = before - after
    print(f"  Memories: {before} → {after}", end="")
    if delta > 0:
        print(f" ({delta} died or consolidated)")
    else:
        print()
    print()

    show(mind, "After reflection")
    print(f"  IDENTITY: {identity.summary}")
    print(f"  TRAITS: {dict(sorted(identity.traits.items(), key=lambda x: -x[1])[:5])}\n")

    # ── Week 6: Deep decay ─────────────────────────────────────────────

    header("WEEK 6 — Only the essential survives")

    time.sleep(5)

    before = mind.count
    identity = mind.reflect()
    after = mind.count

    delta = before - after
    print(f"  Memories: {before} → {after}", end="")
    if delta > 0:
        print(f" ({delta} forgotten)")
    else:
        print()
    print()

    show(mind, "Survivors")
    print(f"  FINAL IDENTITY: {identity.summary}")
    print(f"  RELATIONSHIPS: {identity.relationships}\n")

    # ── Summary ────────────────────────────────────────────────────────

    header("WHAT HAPPENED")

    print("""  DECAY       "good morning" and "weather" faded to nothing.
              Nobody recalled them. They died naturally.

  REINFORCE   "complexity" and "python" survived — because
              we actively thought about them. Use = life.

  CONSOLIDATE "prefers concise and direct answers" merged with
              "prefers concise and direct responses" → one memory.

  EMERGE      Identity was never configured. It was computed
              from the pattern of what survived.

  ONE FILE    The entire mind = one .db file. Portable. Copyable.

  ─────────────────────────────────────────────────────────────
  Zero dependencies. No LLM. No API keys.

  Add kore-bridge for LLM-powered cognition:
    pip install kore-bridge[openai]
""")

    mind.close()
    os.unlink(db_path)


if __name__ == "__main__":
    main()

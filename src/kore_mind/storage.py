"""SQLite storage. Un archivo = una mente."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from kore_mind.models import Identity, Memory, MemoryType


class Storage:
    """SQLite backend. Zero config. Portable."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'episodic',
                salience REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS identity (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memories_salience
                ON memories(salience DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(type);
            CREATE INDEX IF NOT EXISTS idx_memories_created
                ON memories(created_at DESC);
        """)
        self.conn.commit()

    def save_memory(self, mem: Memory) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, content, type, salience, created_at, last_accessed,
                access_count, source, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mem.id, mem.content, mem.type.value, mem.salience,
                mem.created_at, mem.last_accessed, mem.access_count,
                mem.source, json.dumps(mem.tags),
            ),
        )
        self.conn.commit()

    def load_memory(self, memory_id: str) -> Memory | None:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def all_memories(self, min_salience: float = 0.0) -> list[Memory]:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE salience >= ? ORDER BY salience DESC",
            (min_salience,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def top_memories(self, limit: int = 20, min_salience: float = 0.0) -> list[Memory]:
        rows = self.conn.execute(
            """SELECT * FROM memories WHERE salience >= ?
               ORDER BY salience DESC, last_accessed DESC
               LIMIT ?""",
            (min_salience, limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def memories_by_type(self, mem_type: MemoryType) -> list[Memory]:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE type = ? ORDER BY salience DESC",
            (mem_type.value,),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def delete_memory(self, memory_id: str) -> bool:
        cursor = self.conn.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_below_salience(self, threshold: float) -> int:
        cursor = self.conn.execute(
            "DELETE FROM memories WHERE salience < ?", (threshold,)
        )
        self.conn.commit()
        return cursor.rowcount

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def save_identity(self, identity: Identity) -> None:
        data = {
            "traits": json.dumps(identity.traits),
            "summary": identity.summary,
            "relationships": json.dumps(identity.relationships),
        }
        for key, value in data.items():
            self.conn.execute(
                """INSERT OR REPLACE INTO identity (key, value, updated_at)
                   VALUES (?, ?, ?)""",
                (key, value, identity.updated_at),
            )
        self.conn.commit()

    def load_identity(self) -> Identity:
        rows = self.conn.execute("SELECT key, value, updated_at FROM identity").fetchall()
        if not rows:
            return Identity()
        data = {r[0]: r[1] for r in rows}
        updated = max(r[2] for r in rows)
        return Identity(
            traits=json.loads(data.get("traits", "{}")),
            summary=data.get("summary", ""),
            relationships=json.loads(data.get("relationships", "{}")),
            updated_at=updated,
        )

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def _row_to_memory(row: tuple) -> Memory:
        return Memory(
            id=row[0],
            content=row[1],
            type=MemoryType(row[2]),
            salience=row[3],
            created_at=row[4],
            last_accessed=row[5],
            access_count=row[6],
            source=row[7],
            tags=json.loads(row[8]),
        )

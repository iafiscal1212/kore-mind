"""SQLite storage. Un archivo = una mente."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from kore_mind.models import CacheEntry, Identity, Memory, MemoryType, Trace


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
                tags TEXT NOT NULL DEFAULT '[]',
                embedding BLOB
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
            CREATE INDEX IF NOT EXISTS idx_memories_source
                ON memories(source);

            CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                input_text TEXT NOT NULL DEFAULT '',
                output_text TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                duration_ms REAL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_traces_operation
                ON traces(operation);
            CREATE INDEX IF NOT EXISTS idx_traces_created
                ON traces(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_traces_source
                ON traces(source);

            CREATE TABLE IF NOT EXISTS cache (
                id TEXT PRIMARY KEY,
                query_hash TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                ttl REAL NOT NULL DEFAULT 3600,
                hit_count INTEGER NOT NULL DEFAULT 0,
                embedding BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_cache_hash
                ON cache(query_hash);

            CREATE TABLE IF NOT EXISTS query_log (
                id TEXT PRIMARY KEY,
                query_hash TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_query_log_hash
                ON query_log(query_hash, source);
            CREATE INDEX IF NOT EXISTS idx_query_log_created
                ON query_log(created_at);
        """)
        # Migration: add embedding column if upgrading from v0.1.0
        try:
            self.conn.execute("SELECT embedding FROM memories LIMIT 0")
        except sqlite3.OperationalError:
            self.conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
        self.conn.commit()

    # ── Memory CRUD ────────────────────────────────────────────────────

    def save_memory(self, mem: Memory) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, content, type, salience, created_at, last_accessed,
                access_count, source, tags, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mem.id, mem.content, mem.type.value, mem.salience,
                mem.created_at, mem.last_accessed, mem.access_count,
                mem.source, json.dumps(mem.tags), mem.embedding,
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

    def top_memories(self, limit: int = 20, min_salience: float = 0.0,
                     source: str | None = None) -> list[Memory]:
        if source is not None:
            rows = self.conn.execute(
                """SELECT * FROM memories WHERE salience >= ? AND source = ?
                   ORDER BY salience DESC, last_accessed DESC
                   LIMIT ?""",
                (min_salience, source, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM memories WHERE salience >= ?
                   ORDER BY salience DESC, last_accessed DESC
                   LIMIT ?""",
                (min_salience, limit),
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def memories_by_source(self, source: str,
                           min_salience: float = 0.0) -> list[Memory]:
        rows = self.conn.execute(
            """SELECT * FROM memories WHERE source = ? AND salience >= ?
               ORDER BY salience DESC""",
            (source, min_salience),
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

    # ── Traces ─────────────────────────────────────────────────────────

    def save_trace(self, trace: Trace) -> None:
        self.conn.execute(
            """INSERT INTO traces
               (id, operation, input_text, output_text, source,
                duration_ms, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace.id, trace.operation, trace.input_text,
                trace.output_text, trace.source, trace.duration_ms,
                json.dumps(trace.metadata), trace.created_at,
            ),
        )
        self.conn.commit()

    def load_traces(self, operation: str | None = None,
                    source: str | None = None,
                    limit: int = 100) -> list[Trace]:
        query = "SELECT * FROM traces WHERE 1=1"
        params: list = []
        if operation is not None:
            query += " AND operation = ?"
            params.append(operation)
        if source is not None:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_trace(r) for r in rows]

    # ── Cache ──────────────────────────────────────────────────────────

    def save_cache_entry(self, entry: CacheEntry) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO cache
               (id, query_hash, query, response, source,
                created_at, ttl, hit_count, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id, entry.query_hash, entry.query,
                entry.response, entry.source, entry.created_at,
                entry.ttl, entry.hit_count, entry.embedding,
            ),
        )
        self.conn.commit()

    def find_cache_by_hash(self, query_hash: str,
                           source: str | None = None) -> CacheEntry | None:
        if source is not None:
            row = self.conn.execute(
                """SELECT * FROM cache
                   WHERE query_hash = ? AND source = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (query_hash, source),
            ).fetchone()
        else:
            row = self.conn.execute(
                """SELECT * FROM cache
                   WHERE query_hash = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (query_hash,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_cache_entry(row)

    def cache_hit(self, entry_id: str) -> None:
        self.conn.execute(
            "UPDATE cache SET hit_count = hit_count + 1 WHERE id = ?",
            (entry_id,),
        )
        self.conn.commit()

    def delete_expired_cache(self) -> int:
        now = time.time()
        cursor = self.conn.execute(
            "DELETE FROM cache WHERE created_at + ttl < ?", (now,),
        )
        self.conn.commit()
        return cursor.rowcount

    # ── Query Log (rate limiting) ──────────────────────────────────────

    def log_query(self, query_hash: str, source: str = "") -> None:
        import uuid
        self.conn.execute(
            "INSERT INTO query_log (id, query_hash, source, created_at) VALUES (?, ?, ?, ?)",
            (uuid.uuid4().hex[:12], query_hash, source, time.time()),
        )
        self.conn.commit()

    def query_count(self, query_hash: str, source: str = "",
                    window: float = 3600.0) -> int:
        since = time.time() - window
        row = self.conn.execute(
            """SELECT COUNT(*) FROM query_log
               WHERE query_hash = ? AND source = ? AND created_at >= ?""",
            (query_hash, source, since),
        ).fetchone()
        return row[0]

    def cleanup_query_log(self, max_age: float = 86400.0) -> int:
        cutoff = time.time() - max_age
        cursor = self.conn.execute(
            "DELETE FROM query_log WHERE created_at < ?", (cutoff,),
        )
        self.conn.commit()
        return cursor.rowcount

    # ── Close ──────────────────────────────────────────────────────────

    def close(self) -> None:
        self.conn.close()

    # ── Row mappers ────────────────────────────────────────────────────

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
            embedding=row[9] if len(row) > 9 else None,
        )

    @staticmethod
    def _row_to_trace(row: tuple) -> Trace:
        return Trace(
            id=row[0],
            operation=row[1],
            input_text=row[2],
            output_text=row[3],
            source=row[4],
            duration_ms=row[5],
            metadata=json.loads(row[6]),
            created_at=row[7],
        )

    @staticmethod
    def _row_to_cache_entry(row: tuple) -> CacheEntry:
        return CacheEntry(
            id=row[0],
            query_hash=row[1],
            query=row[2],
            response=row[3],
            source=row[4],
            created_at=row[5],
            ttl=row[6],
            hit_count=row[7],
            embedding=row[8],
        )

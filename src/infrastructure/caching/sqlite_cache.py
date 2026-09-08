"""SQLite-backed persistent file cache implementation."""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from src.core.interfaces.cache import BaseCache

logger = logging.getLogger("AskMyPDF.Cache")


class SQLiteDiskCache(BaseCache):
    """File-based persistent cache using SQLite for ACID compliance and hot-swappability."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_store (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    access_count INTEGER DEFAULT 0,
                    PRIMARY KEY (namespace, key)
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expiry ON cache_store(expires_at);")
            conn.commit()

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Retrieve a cached value by namespace and key, validating expiration."""
        now = time.time()
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value_json, expires_at FROM cache_store WHERE namespace = ? AND key = ?",
                (namespace, key)
            )
            row = cursor.fetchone()
            if not row:
                self._misses += 1
                return None

            value_json, expires_at = row
            if expires_at is not None and expires_at < now:
                # Expired: remove it lazily
                cursor.execute(
                    "DELETE FROM cache_store WHERE namespace = ? AND key = ?",
                    (namespace, key)
                )
                conn.commit()
                self._misses += 1
                return None

            # Increment access count
            cursor.execute(
                "UPDATE cache_store SET access_count = access_count + 1 WHERE namespace = ? AND key = ?",
                (namespace, key)
            )
            conn.commit()
            self._hits += 1

            try:
                return json.loads(value_json)
            except Exception as e:
                logger.error("Failed to deserialize cached JSON for key '%s': %s", key, e)
                return None

    def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a JSON-serializable value with optional TTL in seconds."""
        now = time.time()
        expires_at = (now + ttl) if ttl else None
        value_json = json.dumps(value)

        with self._lock, self._get_connection() as conn:
            conn.execute("""
                INSERT INTO cache_store (namespace, key, value_json, created_at, expires_at, access_count)
                VALUES (?, ?, ?, ?, ?, 0)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at;
            """, (namespace, key, value_json, now, expires_at))
            conn.commit()

    def delete(self, namespace: str, key: str) -> bool:
        """Delete an item from the cache."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM cache_store WHERE namespace = ? AND key = ?",
                (namespace, key)
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear cache entries, optionally constrained to a single namespace."""
        with self._lock, self._get_connection() as conn:
            if namespace:
                conn.execute("DELETE FROM cache_store WHERE namespace = ?", (namespace,))
            else:
                conn.execute("DELETE FROM cache_store;")
            conn.commit()
            logger.info("Cleared cache for namespace='%s'", namespace or "ALL")

    def get_stats(self) -> Dict[str, Any]:
        """Return cache health metrics and total entries."""
        now = time.time()
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT namespace, COUNT(*) FROM cache_store GROUP BY namespace")
            counts_by_ns = dict(cursor.fetchall())

            total_entries = sum(counts_by_ns.values())
            file_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0

            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0

            return {
                "total_entries": total_entries,
                "namespaces": counts_by_ns,
                "file_size_bytes": file_size_bytes,
                "file_size_kb": round(file_size_bytes / 1024, 2),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percentage": round(hit_rate * 100, 2),
                "db_path": str(self.db_path)
            }

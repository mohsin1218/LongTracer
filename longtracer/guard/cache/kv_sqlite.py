"""
SQLite key-value cache backend (fallback).

Multi-process safe via WAL mode, busy_timeout, and thread-local connections.
Stores values as JSON strings with Unix epoch TTL.
"""

import os
import sqlite3
import threading
import time as _time
import logging
from datetime import datetime
from typing import Callable, Optional

from .kv_backend import CacheBackend

logger = logging.getLogger(__name__)

_RETRY_DELAYS_MS = (50, 100, 200)  # exponential backoff on "database is locked"


class SQLiteCacheBackend(CacheBackend):
    """
    SQLite-backed key-value cache.

    - Path: ``./.cache/cache.sqlite`` by default (auto-creates directory)
    - WAL journal mode + ``PRAGMA busy_timeout = 5000``
    - Thread-local connections via ``threading.local()``
    - Write retry loop (3×, 50 → 100 → 200 ms backoff)
    - TTL via ``now_fn().timestamp()`` — fully clock-injectable

    No external dependencies (sqlite3 is in Python stdlib).
    """

    def __init__(
        self,
        path: Optional[str] = None,
        cleanup_interval: int = 100,
        now_fn: Callable[[], datetime] = datetime.utcnow,
    ):
        self._path = path or os.environ.get(
            "CACHE_SQLITE_PATH",
            os.path.join(os.path.expanduser("~"), ".longtracer", "cache.sqlite"),
        )
        self._cleanup_interval = cleanup_interval
        self._get_counter = 0
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._connected = False

        super().__init__(now_fn=now_fn)
        self._init_db()

    # ── properties ──────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        return "sqlite"

    # ── connection management ───────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local connection, creating one if needed."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, timeout=5, check_same_thread=True)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA busy_timeout = 5000")
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        """Create directory, database, and table."""
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)

            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kv_cache (
                    namespace TEXT NOT NULL,
                    key       TEXT NOT NULL,
                    value     TEXT NOT NULL,
                    expires_at REAL,
                    PRIMARY KEY (namespace, key)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kv_expires ON kv_cache(expires_at)"
            )
            conn.commit()
            self._connected = True
            print(f"  ✅ Cache backend: SQLite [Path: {self._path}]")
        except Exception as exc:
            logger.error("SQLite cache init failed: %s", exc)
            self._connected = False

    # ── retry helper ────────────────────────────────────────────────

    def _execute_write(self, conn: sqlite3.Connection, sql: str, params: tuple) -> sqlite3.Cursor:
        """Execute a write with retry on 'database is locked'."""
        for attempt, delay_ms in enumerate(_RETRY_DELAYS_MS):
            try:
                with self._write_lock:
                    cur = conn.execute(sql, params)
                    conn.commit()
                    return cur
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and attempt < len(_RETRY_DELAYS_MS) - 1:
                    _time.sleep(delay_ms / 1000)
                    continue
                raise
        # unreachable, but keeps type checkers happy
        raise sqlite3.OperationalError("database is locked (retries exhausted)")

    # ── periodic cleanup ────────────────────────────────────────────

    def _maybe_cleanup(self) -> None:
        self._get_counter += 1
        if self._get_counter % self._cleanup_interval == 0:
            now_ts = self._now_fn().timestamp()
            try:
                conn = self._get_conn()
                with self._write_lock:
                    conn.execute(
                        "DELETE FROM kv_cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                        (now_ts,),
                    )
                    conn.commit()
            except Exception:
                pass  # cleanup is best-effort

    # ── interface implementation ────────────────────────────────────

    def _get(self, key: str, namespace: str) -> Optional[str]:
        if not self._connected:
            return None

        now_ts = self._now_fn().timestamp()
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM kv_cache "
            "WHERE namespace = ? AND key = ? "
            "AND (expires_at IS NULL OR expires_at > ?)",
            (namespace, key, now_ts),
        ).fetchone()

        self._maybe_cleanup()
        return row["value"] if row else None

    def _set(
        self,
        key: str,
        value_json: str,
        namespace: str,
        expires_at: Optional[datetime],
    ) -> None:
        if not self._connected:
            return

        ea_ts = expires_at.timestamp() if expires_at else None
        conn = self._get_conn()
        self._execute_write(
            conn,
            "INSERT OR REPLACE INTO kv_cache (namespace, key, value, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (namespace, key, value_json, ea_ts),
        )

    def _delete(self, key: str, namespace: str) -> bool:
        if not self._connected:
            return False

        conn = self._get_conn()
        cur = self._execute_write(
            conn,
            "DELETE FROM kv_cache WHERE namespace = ? AND key = ?",
            (namespace, key),
        )
        return cur.rowcount > 0

    def clear_namespace(self, namespace: str) -> int:
        if not self._connected:
            return 0

        conn = self._get_conn()
        cur = self._execute_write(
            conn,
            "DELETE FROM kv_cache WHERE namespace = ?",
            (namespace,),
        )
        return cur.rowcount

    def is_connected(self) -> bool:
        return self._connected

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None
        self._connected = False

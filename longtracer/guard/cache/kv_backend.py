"""
Abstract interface for pluggable key-value cache backends.

Provides a general-purpose cache (separate from TraceCacheBackend)
with TTL support, namespace partitioning, and observability.

Serialization contract:
    - Values must be JSON-serializable (dict/list/str/int/float/bool/None)
    - set() raises TypeError for non-serializable, ValueError for NaN/Inf
    - bytes must be base64-encoded by the caller before storing
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional


# ── Serialization helpers ───────────────────────────────────────────

try:
    import orjson as _json_lib

    def _dumps(value: Any) -> str:
        """Serialize to JSON string (orjson — strict, fast)."""
        return _json_lib.dumps(value).decode("utf-8")

    def _loads(raw: str) -> Any:
        return _json_lib.loads(raw)

except ImportError:
    _json_lib = None  # type: ignore[assignment]

    def _dumps(value: Any) -> str:  # type: ignore[misc]
        """Serialize to JSON string (stdlib — allow_nan=False)."""
        return json.dumps(value, allow_nan=False)

    def _loads(raw: str) -> Any:  # type: ignore[misc]
        return json.loads(raw)


# ── Observability ───────────────────────────────────────────────────

@dataclass
class CacheStats:
    """In-process cache hit/miss and latency counters (not persisted)."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    _total_get_ms: float = field(default=0.0, repr=False)
    _get_count: int = field(default=0, repr=False)
    backend: str = ""

    @property
    def avg_get_ms(self) -> float:
        return self._total_get_ms / self._get_count if self._get_count else 0.0

    def record_get(self, hit: bool, elapsed_ms: float) -> None:
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        self._total_get_ms += elapsed_ms
        self._get_count += 1

    def snapshot(self) -> "CacheStats":
        """Return a frozen copy for external consumption."""
        s = CacheStats(
            hits=self.hits,
            misses=self.misses,
            sets=self.sets,
            deletes=self.deletes,
            errors=self.errors,
            backend=self.backend,
        )
        s._total_get_ms = self._total_get_ms
        s._get_count = self._get_count
        return s


# ── Key helper ──────────────────────────────────────────────────────

def cache_key(*parts: str) -> str:
    """
    Build a stable, fixed-length cache key from arbitrary parts.

    Usage:
        key = cache_key("user", user_id, "profile")
        # → sha256 hex digest (64 chars)

    Use for composite or potentially-large keys.
    For short, readable keys you can pass a plain string directly.
    """
    raw = ":".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Abstract backend ────────────────────────────────────────────────

class CacheBackend(ABC):
    """
    Abstract base for key-value cache backends.

    All concrete backends (MongoDB, SQLite) implement these methods.
    Namespace is always a method argument — never stored on the instance.

    TTL semantics:
        ttl_seconds=None  → never expires
        ttl_seconds=0     → not stored (get always returns None)
        ttl_seconds>0     → expires after N seconds
    """

    def __init__(self, now_fn: Callable[[], datetime] = datetime.utcnow):
        """
        Args:
            now_fn: Clock function for TTL. Override in tests for
                    deterministic expiry without sleeping.
        """
        self._now_fn = now_fn
        self._stats = CacheStats(backend=self.backend_name)

    # ── Interface ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier, e.g. 'mongodb' or 'sqlite'."""
        ...

    @abstractmethod
    def _get(self, key: str, namespace: str) -> Optional[str]:
        """Return raw JSON string or None. Implementors handle TTL filtering."""
        ...

    @abstractmethod
    def _set(self, key: str, value_json: str, namespace: str,
             expires_at: Optional[datetime]) -> None:
        """Store raw JSON string with optional expiry."""
        ...

    @abstractmethod
    def _delete(self, key: str, namespace: str) -> bool:
        """Delete a single key. Return True if it existed."""
        ...

    @abstractmethod
    def clear_namespace(self, namespace: str) -> int:
        """Delete all keys in a namespace. Return count deleted."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the backend is operational."""
        ...

    # ── Public API (wraps _get/_set/_delete with stats + serde) ─────

    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """
        Retrieve a value by key and namespace.

        Returns None on miss or expired entry.
        """
        t0 = time.monotonic()
        try:
            raw = self._get(key, namespace)
        except Exception:
            self._stats.errors += 1
            raise
        elapsed = (time.monotonic() - t0) * 1000

        if raw is None:
            self._stats.record_get(hit=False, elapsed_ms=elapsed)
            return None

        self._stats.record_get(hit=True, elapsed_ms=elapsed)
        return _loads(raw)

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        namespace: str = "default",
    ) -> None:
        """
        Store a value.

        Raises:
            TypeError:  value is not JSON-serializable
            ValueError: value contains NaN or Inf floats
        """
        # ttl_seconds=0 → do not store
        if ttl_seconds is not None and ttl_seconds <= 0:
            return

        # Serialize (raises TypeError or ValueError on bad input)
        value_json = _dumps(value)

        # Compute expiry
        expires_at: Optional[datetime] = None
        if ttl_seconds is not None:
            from datetime import timedelta
            expires_at = self._now_fn() + timedelta(seconds=ttl_seconds)

        try:
            self._set(key, value_json, namespace, expires_at)
            self._stats.sets += 1
        except Exception:
            self._stats.errors += 1
            raise

    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a key. Returns True if it existed."""
        try:
            removed = self._delete(key, namespace)
            if removed:
                self._stats.deletes += 1
            return removed
        except Exception:
            self._stats.errors += 1
            raise

    def get_stats(self) -> CacheStats:
        """Return a snapshot of cache statistics."""
        return self._stats.snapshot()

"""
MongoDB key-value cache backend.

Primary backend when MONGODB_URI is configured. Stores values as JSON strings
with BSON datetime TTL (expireAfterSeconds=0). Falls back to SQLite on failure.
"""

import os
import logging
from datetime import datetime
from typing import Callable, Optional

from .kv_backend import CacheBackend

logger = logging.getLogger(__name__)

try:
    from pymongo import MongoClient, ASCENDING
    from pymongo.errors import (
        ConnectionFailure,
        ConfigurationError,
        OperationFailure,
        ServerSelectionTimeoutError,
    )
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


class MongoCacheBackend(CacheBackend):
    """
    MongoDB-backed key-value cache.

    - Collection: ``kv_cache`` with unique compound index ``(namespace, key)``
    - TTL: ``expireAfterSeconds=0`` on ``expires_at`` (BSON datetime)
    - Python-side expiry check in get() as belt-and-suspenders
    - Credential-safe logging: never prints raw URI

    Requires: ``pip install pymongo>=4.6.0``
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        database: Optional[str] = None,
        collection: str = "kv_cache",
        now_fn: Callable[[], datetime] = datetime.utcnow,
    ):
        if not PYMONGO_AVAILABLE:
            raise ImportError(
                "pymongo is not installed. Install with: pip install pymongo>=4.6.0"
            )

        self._uri = uri or os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
        self._database_name = database or os.environ.get("MONGODB_DATABASE", "longtracer")
        self._collection_name = collection
        self._client = None
        self._col = None
        self._connected = False

        # Must set backend_name before super().__init__ accesses it
        super().__init__(now_fn=now_fn)
        self._connect()

    # ── properties ──────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        return "mongodb"

    # ── connection ──────────────────────────────────────────────────

    def _connect(self) -> None:
        try:
            self._client = MongoClient(
                self._uri,
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=3000,
            )
            self._client.admin.command("ping")

            db = self._client[self._database_name]
            self._col = db[self._collection_name]

            # Compound unique index
            self._col.create_index(
                [("namespace", ASCENDING), ("key", ASCENDING)],
                unique=True,
            )
            # TTL index — MongoDB auto-deletes docs when expires_at passes
            self._col.create_index("expires_at", expireAfterSeconds=0)

            self._connected = True
            print(
                f"  ✅ Cache backend: MongoDB "
                f"[DB: {self._database_name}, collection: {self._collection_name}]"
            )
        except (
            ConnectionFailure,
            ServerSelectionTimeoutError,
            ConfigurationError,
            OperationFailure,
        ) as exc:
            exc_name = type(exc).__name__
            logger.warning(
                "MongoDB cache unavailable [DB: %s] (%s): %s",
                self._database_name, exc_name, exc,
            )
            self._connected = False

    # ── interface implementation ────────────────────────────────────

    def _get(self, key: str, namespace: str) -> Optional[str]:
        if not self._connected:
            return None

        now = self._now_fn()
        doc = self._col.find_one(
            {"namespace": namespace, "key": key},
            {"value": 1, "expires_at": 1, "_id": 0},
        )
        if doc is None:
            return None

        # Belt-and-suspenders: check expiry in Python too (TTL lag ~60 s)
        ea = doc.get("expires_at")
        if ea is not None and ea <= now:
            return None

        return doc.get("value")

    def _set(
        self,
        key: str,
        value_json: str,
        namespace: str,
        expires_at: Optional[datetime],
    ) -> None:
        if not self._connected:
            return

        doc = {
            "namespace": namespace,
            "key": key,
            "value": value_json,
            "created_at": self._now_fn(),
        }
        if expires_at is not None:
            doc["expires_at"] = expires_at
        else:
            doc["expires_at"] = None  # no TTL

        self._col.update_one(
            {"namespace": namespace, "key": key},
            {"$set": doc},
            upsert=True,
        )

    def _delete(self, key: str, namespace: str) -> bool:
        if not self._connected:
            return False
        result = self._col.delete_one({"namespace": namespace, "key": key})
        return result.deleted_count > 0

    def clear_namespace(self, namespace: str) -> int:
        if not self._connected:
            return 0
        result = self._col.delete_many({"namespace": namespace})
        return result.deleted_count

    def is_connected(self) -> bool:
        return self._connected

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._connected = False

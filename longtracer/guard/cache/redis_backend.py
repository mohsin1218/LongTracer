"""
Redis cache backend for distributed caching.

Fast, distributed cache. Good for multi-worker setups and horizontal scaling.
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from .backend import TraceCacheBackend

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisBackend(TraceCacheBackend):
    """
    Redis-based trace storage backend.
    
    Use cases:
    - Distributed systems with multiple workers
    - High-throughput environments
    - Fast cache with automatic expiry
    - Horizontal scaling
    
    Requires: pip install redis>=4.0.0
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "trace:",
        ttl_seconds: Optional[int] = None  # None = no expiry
    ):
        """
        Initialize Redis backend.
        
        Args:
            url: Redis connection URL (e.g., redis://localhost:6379/0)
            host: Redis host (default: REDIS_HOST env var or localhost)
            port: Redis port (default: REDIS_PORT env var or 6379)
            db: Redis database number
            password: Redis password (default: REDIS_PASSWORD env var)
            prefix: Key prefix for all trace keys
            ttl_seconds: Time-to-live for traces (None = no expiry)
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis dependencies not installed. "
                "Install with: pip install redis>=4.0.0"
            )
        
        self._prefix = prefix
        self._ttl = ttl_seconds
        self._connected = False
        
        try:
            if url:
                self._client = redis.from_url(url)
            else:
                self._client = redis.Redis(
                    host=host or os.environ.get("REDIS_HOST", "localhost"),
                    port=port or int(os.environ.get("REDIS_PORT", 6379)),
                    db=db,
                    password=password or os.environ.get("REDIS_PASSWORD"),
                    decode_responses=True
                )
            
            # Test connection
            self._client.ping()
            self._connected = True
            print(f"  ✅ Redis Backend Active [prefix: {prefix}]")
            
        except Exception as e:
            print(f"  ⚠️  Redis connection failed: {e}")
            self._connected = False
    
    def _key(self, key_type: str, id: str) -> str:
        """Generate prefixed key."""
        return f"{self._prefix}{key_type}:{id}"
    
    def _serialize(self, data: Dict[str, Any]) -> str:
        """Serialize dict to JSON string."""
        return json.dumps(data, default=str)
    
    def _deserialize(self, data: str) -> Dict[str, Any]:
        """Deserialize JSON string to dict."""
        return json.loads(data) if data else {}
    
    def save_run(self, run: Dict[str, Any]) -> str:
        """Save run to Redis."""
        if not self._connected:
            return run.get("run_id", "")
        
        try:
            run_id = run.get("run_id", "")
            trace_id = run.get("trace_id", "")
            
            # Save run data
            key = self._key("run", run_id)
            self._client.set(key, self._serialize(run))
            if self._ttl:
                self._client.expire(key, self._ttl)
            
            # Add to trace's run list
            trace_runs_key = self._key("trace_runs", trace_id)
            self._client.rpush(trace_runs_key, run_id)
            if self._ttl:
                self._client.expire(trace_runs_key, self._ttl)
            
            return run_id
        except Exception as e:
            print(f"  ⚠️  Failed to save run to Redis: {e}")
            return run.get("run_id", "")
    
    def update_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Update run in Redis."""
        if not self._connected:
            return False
        
        try:
            key = self._key("run", run_id)
            data = self._client.get(key)
            if not data:
                return False
            
            run = self._deserialize(data)
            run.update(updates)
            self._client.set(key, self._serialize(run))
            return True
        except Exception as e:
            print(f"  ⚠️  Failed to update run in Redis: {e}")
            return False
    
    def save_trace(self, trace: Dict[str, Any]) -> str:
        """Save trace to Redis."""
        if not self._connected:
            return trace.get("trace_id", "")
        
        try:
            trace_id = trace.get("trace_id", "")
            
            # Add timestamp for ordering
            if "created_at" not in trace:
                trace["created_at"] = datetime.utcnow().isoformat()
            elif isinstance(trace["created_at"], datetime):
                trace["created_at"] = trace["created_at"].isoformat()
            
            # Save trace data
            key = self._key("trace", trace_id)
            self._client.set(key, self._serialize(trace))
            if self._ttl:
                self._client.expire(key, self._ttl)
            
            # Add to recent traces sorted set (score = timestamp)
            timestamp = datetime.utcnow().timestamp()
            self._client.zadd(self._key("traces", "recent"), {trace_id: timestamp})
            
            return trace_id
        except Exception as e:
            print(f"  ⚠️  Failed to save trace to Redis: {e}")
            return trace.get("trace_id", "")
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve trace from Redis."""
        if not self._connected:
            return None
        
        try:
            key = self._key("trace", trace_id)
            data = self._client.get(key)
            return self._deserialize(data) if data else None
        except Exception as e:
            print(f"  ⚠️  Failed to get trace from Redis: {e}")
            return None
    
    def list_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent traces from Redis (newest first)."""
        if not self._connected:
            return []
        
        try:
            # Get recent trace IDs (sorted by timestamp, descending)
            trace_ids = self._client.zrevrange(
                self._key("traces", "recent"), 0, limit - 1
            )
            
            traces = []
            for trace_id in trace_ids:
                trace = self.get_trace(trace_id)
                if trace:
                    traces.append(trace)
            
            return traces
        except Exception as e:
            print(f"  ⚠️  Failed to list traces from Redis: {e}")
            return []
    
    def get_runs_by_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all runs for a trace from Redis."""
        if not self._connected:
            return []
        
        try:
            trace_runs_key = self._key("trace_runs", trace_id)
            run_ids = self._client.lrange(trace_runs_key, 0, -1)
            
            runs = []
            for run_id in run_ids:
                key = self._key("run", run_id)
                data = self._client.get(key)
                if data:
                    runs.append(self._deserialize(data))
            
            return runs
        except Exception as e:
            print(f"  ⚠️  Failed to get runs from Redis: {e}")
            return []
    
    def is_connected(self) -> bool:
        """Check Redis connection status."""
        return self._connected
    
    def close(self):
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._connected = False

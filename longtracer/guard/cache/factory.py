"""
Factory for creating trace cache backends.

Usage:
    from longtracer.guard.cache import create_backend
    
    # From environment variable TRACE_CACHE_BACKEND
    backend = create_backend()
    
    # Explicit backend type
    backend = create_backend("mongo", uri="mongodb://localhost:27017")
    backend = create_backend("sqlite", path="./traces.db")
    backend = create_backend("redis", host="localhost", port=6379)
    backend = create_backend("postgres", database="mydb")
    backend = create_backend("memory")
"""

import os
from typing import Optional

from .backend import TraceCacheBackend


def create_backend(
    backend_type: Optional[str] = None,
    **kwargs
) -> TraceCacheBackend:
    """
    Factory to create cache backend from configuration.
    
    Args:
        backend_type: Type of backend. Options:
            - "sqlite" - SQLite file database (default, ~/.longtracer/traces.db)
            - "memory" / "mem" - In-memory (for testing, lost on restart)
            - "mongo" / "mongodb" - MongoDB
            - "redis" - Redis distributed cache
            - "postgres" / "postgresql" - PostgreSQL
        **kwargs: Backend-specific arguments passed to constructor.
        
    Returns:
        Configured TraceCacheBackend instance.
        
    Raises:
        ValueError: If backend_type is unknown.
        
    Examples:
        # Use environment variable
        backend = create_backend()
        
        # MongoDB
        backend = create_backend("mongo", uri="mongodb://localhost:27017", database="my_db")
        
        # SQLite
        backend = create_backend("sqlite", path="./my_traces.db")
        
        # Redis
        backend = create_backend("redis", host="localhost", port=6379, ttl_seconds=3600)
        
        # PostgreSQL
        backend = create_backend("postgres", host="localhost", database="traces")
        
        # In-memory for testing
        backend = create_backend("memory", max_traces=500)
    """
    backend_type = backend_type or os.environ.get("TRACE_CACHE_BACKEND", "sqlite")
    backend_type = backend_type.lower().strip()
    
    if backend_type == "mongo" or backend_type == "mongodb":
        from .mongo import MongoBackend
        return MongoBackend(**kwargs)
    
    elif backend_type == "sqlite":
        from .sqlite import SQLiteBackend
        return SQLiteBackend(**kwargs)
    
    elif backend_type == "redis":
        from .redis_backend import RedisBackend
        return RedisBackend(**kwargs)
    
    elif backend_type == "postgres" or backend_type == "postgresql":
        from .postgres import PostgresBackend
        return PostgresBackend(**kwargs)
    
    elif backend_type == "memory" or backend_type == "mem":
        from .memory import MemoryBackend
        return MemoryBackend(**kwargs)
    
    else:
        valid_backends = [
            "memory", "mem", 
            "sqlite", 
            "mongo", "mongodb", 
            "redis", 
            "postgres", "postgresql"
        ]
        raise ValueError(
            f"Unknown backend type: '{backend_type}'. "
            f"Valid options: {valid_backends}"
        )


def get_default_backend() -> TraceCacheBackend:
    """
    Get the default backend based on environment.
    
    Priority:
    1. TRACE_CACHE_BACKEND env var
    2. If MONGODB_URI is set, use mongo
    3. If REDIS_HOST is set, use redis
    4. If POSTGRES_HOST is set, use postgres
    5. Otherwise, use sqlite (~/.longtracer/traces.db)
    """
    explicit = os.environ.get("TRACE_CACHE_BACKEND")
    if explicit:
        return create_backend(explicit)
    
    if os.environ.get("MONGODB_URI"):
        return create_backend("mongo")
    
    if os.environ.get("REDIS_HOST"):
        return create_backend("redis")
    
    if os.environ.get("POSTGRES_HOST"):
        return create_backend("postgres")
    
    return create_backend("sqlite")


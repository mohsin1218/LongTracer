"""
MongoDB cache backend for production trace storage.

Extracted and refactored from mongo_tracer.py to implement TraceCacheBackend interface.
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from .backend import TraceCacheBackend

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


class MongoBackend(TraceCacheBackend):
    """
    MongoDB-based trace storage backend.
    
    Use cases:
    - Production environments
    - Distributed systems (multiple workers sharing traces)
    - Long-term trace persistence and querying
    
    Requires: pip install pymongo>=4.6.0
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        database: Optional[str] = None,
        traces_collection: str = "traces",
        runs_collection: str = "runs"
    ):
        """
        Initialize MongoDB backend.
        
        Args:
            uri: MongoDB connection URI (default: MONGODB_URI env var)
            database: Database name (default: MONGODB_DATABASE env var or "longtracer")
            traces_collection: Collection name for traces
            runs_collection: Collection name for runs
        """
        if not PYMONGO_AVAILABLE:
            raise ImportError(
                "MongoDB dependencies not installed. "
                "Install with: pip install pymongo>=4.6.0"
            )
        
        self._uri = uri or os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
        self._database_name = database or os.environ.get("MONGODB_DATABASE", "longtracer")
        self._traces_collection_name = traces_collection or os.environ.get("MONGODB_COLLECTION_TRACES", "traces")
        self._runs_collection_name = runs_collection or os.environ.get("MONGODB_COLLECTION_RUNS", "runs")
        
        self._client = None
        self._db = None
        self._traces = None
        self._runs = None
        self._connected = False
        
        self._connect()
    
    def _connect(self):
        """Establish MongoDB connection and create indexes."""
        try:
            self._client = MongoClient(
                self._uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            # Test connection
            self._client.admin.command('ping')
            
            self._db = self._client[self._database_name]
            self._traces = self._db[self._traces_collection_name]
            self._runs = self._db[self._runs_collection_name]
            
            # Create indexes for query performance
            self._traces.create_index("trace_id")
            self._traces.create_index("project_name")
            self._traces.create_index("created_at")
            self._runs.create_index("run_id")
            self._runs.create_index("trace_id")
            self._runs.create_index("parent_id")
            
            self._connected = True
            print(f"  ✅ MongoDB Backend Active [DB: {self._database_name}]")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"  ⚠️  MongoDB connection failed: {e}")
            print(f"     URI: {self._uri}")
            self._connected = False
    
    def save_run(self, run: Dict[str, Any]) -> str:
        """Save run to MongoDB."""
        if not self._connected:
            return run.get("run_id", "")
        
        try:
            self._runs.insert_one(run.copy())
            return run.get("run_id", "")
        except Exception as e:
            print(f"  ⚠️  Failed to save run to MongoDB: {e}")
            return run.get("run_id", "")
    
    def update_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Update run in MongoDB."""
        if not self._connected:
            return False
        
        try:
            result = self._runs.update_one(
                {"run_id": run_id},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"  ⚠️  Failed to update run in MongoDB: {e}")
            return False
    
    def save_trace(self, trace: Dict[str, Any]) -> str:
        """Save trace to MongoDB."""
        if not self._connected:
            return trace.get("trace_id", "")
        
        try:
            self._traces.insert_one(trace.copy())
            return trace.get("trace_id", "")
        except Exception as e:
            print(f"  ⚠️  Failed to save trace to MongoDB: {e}")
            return trace.get("trace_id", "")
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve trace from MongoDB."""
        if not self._connected:
            return None
        
        try:
            return self._traces.find_one({"trace_id": trace_id})
        except Exception as e:
            print(f"  ⚠️  Failed to get trace from MongoDB: {e}")
            return None
    
    def list_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent traces from MongoDB."""
        if not self._connected:
            return []
        
        try:
            return list(
                self._traces
                .find()
                .sort("created_at", -1)
                .limit(limit)
            )
        except Exception as e:
            print(f"  ⚠️  Failed to list traces from MongoDB: {e}")
            return []
    
    def get_runs_by_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all runs for a trace from MongoDB."""
        if not self._connected:
            return []
        
        try:
            return list(
                self._runs
                .find({"trace_id": trace_id})
                .sort("created_at", 1)
            )
        except Exception as e:
            print(f"  ⚠️  Failed to get runs from MongoDB: {e}")
            return []
    
    def is_connected(self) -> bool:
        """Check MongoDB connection status."""
        return self._connected
    
    def close(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._connected = False

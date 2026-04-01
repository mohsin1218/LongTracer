"""
Abstract interface for trace cache backends.

All cache backends must implement TraceCacheBackend.
This allows swapping databases (Mongo, SQLite, Memory) without changing application code.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class TraceCacheBackend(ABC):
    """
    Abstract base class for trace storage backends.
    
    All concrete backends (MongoDB, SQLite, Memory) must implement these methods.
    This enables database-agnostic trace persistence.
    """
    
    @abstractmethod
    def save_run(self, run: Dict[str, Any]) -> str:
        """
        Save a run document to the cache.
        
        Args:
            run: Run document with run_id, trace_id, name, inputs, outputs, etc.
            
        Returns:
            The run_id of the saved run.
        """
        pass
    
    @abstractmethod
    def update_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing run with new data.
        
        Args:
            run_id: ID of the run to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update succeeded, False otherwise.
        """
        pass
    
    @abstractmethod
    def save_trace(self, trace: Dict[str, Any]) -> str:
        """
        Save an aggregated trace document.
        
        Args:
            trace: Trace document with trace_id, inputs, outputs, claim_evidence_map, etc.
            
        Returns:
            The trace_id of the saved trace.
        """
        pass
    
    @abstractmethod
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a trace by its ID.
        
        Args:
            trace_id: ID of the trace to retrieve
            
        Returns:
            Trace document or None if not found.
        """
        pass
    
    @abstractmethod
    def list_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent traces, ordered by creation time (newest first).
        
        Args:
            limit: Maximum number of traces to return
            
        Returns:
            List of trace documents.
        """
        pass
    
    @abstractmethod
    def get_runs_by_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """
        Get all runs belonging to a specific trace.
        
        Args:
            trace_id: ID of the trace
            
        Returns:
            List of run documents ordered by creation time.
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the backend is connected and operational.
        
        Returns:
            True if connected, False otherwise.
        """
        pass

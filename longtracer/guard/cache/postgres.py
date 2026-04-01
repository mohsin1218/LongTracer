"""
PostgreSQL cache backend for production trace storage.

Full-featured relational database with ACID compliance, great for production.
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from .backend import TraceCacheBackend

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class PostgresBackend(TraceCacheBackend):
    """
    PostgreSQL-based trace storage backend.
    
    Use cases:
    - Production environments with existing PostgreSQL
    - Need for complex queries and analytics
    - ACID compliance requirements
    - Integration with existing data warehouse
    
    Requires: pip install psycopg2-binary>=2.9.0
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        schema: str = "public"
    ):
        """
        Initialize PostgreSQL backend.
        
        Args:
            url: PostgreSQL connection URL (e.g., postgresql://user:pass@host:5432/db)
            host: PostgreSQL host (default: POSTGRES_HOST env var or localhost)
            port: PostgreSQL port (default: POSTGRES_PORT env var or 5432)
            database: Database name (default: POSTGRES_DB env var or "longtracer")
            user: Database user (default: POSTGRES_USER env var)
            password: Database password (default: POSTGRES_PASSWORD env var)
            schema: Schema to use (default: public)
        """
        if not PSYCOPG2_AVAILABLE:
            raise ImportError(
                "PostgreSQL dependencies not installed. "
                "Install with: pip install psycopg2-binary>=2.9.0"
            )
        
        self._schema = schema
        self._connected = False
        self._conn = None
        
        try:
            if url:
                self._conn = psycopg2.connect(url)
            else:
                self._conn = psycopg2.connect(
                    host=host or os.environ.get("POSTGRES_HOST", "localhost"),
                    port=port or int(os.environ.get("POSTGRES_PORT", 5432)),
                    database=database or os.environ.get("POSTGRES_DB", "longtracer"),
                    user=user or os.environ.get("POSTGRES_USER", "postgres"),
                    password=password or os.environ.get("POSTGRES_PASSWORD", "")
                )
            
            self._conn.autocommit = True
            self._create_tables()
            self._connected = True
            print(f"  ✅ PostgreSQL Backend Active [schema: {schema}]")
            
        except Exception as e:
            print(f"  ⚠️  PostgreSQL connection failed: {e}")
            self._connected = False
    
    def _create_tables(self):
        """Create traces and runs tables if they don't exist."""
        with self._conn.cursor() as cur:
            # Create traces table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._schema}.traces (
                    trace_id VARCHAR(64) PRIMARY KEY,
                    project_name VARCHAR(255),
                    run_name VARCHAR(255),
                    data JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create runs table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._schema}.runs (
                    run_id VARCHAR(64) PRIMARY KEY,
                    trace_id VARCHAR(64),
                    parent_id VARCHAR(64),
                    name VARCHAR(255),
                    data JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create indexes
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_traces_created 
                ON {self._schema}.traces(created_at DESC)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_runs_trace 
                ON {self._schema}.runs(trace_id)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_runs_parent 
                ON {self._schema}.runs(parent_id)
            """)
    
    def save_run(self, run: Dict[str, Any]) -> str:
        """Save run to PostgreSQL."""
        if not self._connected:
            return run.get("run_id", "")
        
        try:
            run_id = run.get("run_id", "")
            trace_id = run.get("trace_id", "")
            parent_id = run.get("parent_id")
            name = run.get("name", "")
            
            with self._conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self._schema}.runs 
                    (run_id, trace_id, parent_id, name, data, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (run_id) DO UPDATE SET
                        data = EXCLUDED.data
                """, (
                    run_id, trace_id, parent_id, name,
                    json.dumps(run, default=str)
                ))
            
            return run_id
        except Exception as e:
            print(f"  ⚠️  Failed to save run to PostgreSQL: {e}")
            return run.get("run_id", "")
    
    def update_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Update run in PostgreSQL."""
        if not self._connected:
            return False
        
        try:
            with self._conn.cursor() as cur:
                # Fetch, merge, update using JSONB operators
                cur.execute(f"""
                    UPDATE {self._schema}.runs 
                    SET data = data || %s::jsonb
                    WHERE run_id = %s
                """, (json.dumps(updates, default=str), run_id))
                
                return cur.rowcount > 0
        except Exception as e:
            print(f"  ⚠️  Failed to update run in PostgreSQL: {e}")
            return False
    
    def save_trace(self, trace: Dict[str, Any]) -> str:
        """Save trace to PostgreSQL."""
        if not self._connected:
            return trace.get("trace_id", "")
        
        try:
            trace_id = trace.get("trace_id", "")
            project_name = trace.get("project_name", "")
            run_name = trace.get("run_name", "")
            
            with self._conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self._schema}.traces 
                    (trace_id, project_name, run_name, data, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (trace_id) DO UPDATE SET
                        data = EXCLUDED.data
                """, (
                    trace_id, project_name, run_name,
                    json.dumps(trace, default=str)
                ))
            
            return trace_id
        except Exception as e:
            print(f"  ⚠️  Failed to save trace to PostgreSQL: {e}")
            return trace.get("trace_id", "")
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve trace from PostgreSQL."""
        if not self._connected:
            return None
        
        try:
            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT data FROM {self._schema}.traces 
                    WHERE trace_id = %s
                """, (trace_id,))
                
                row = cur.fetchone()
                return row["data"] if row else None
        except Exception as e:
            print(f"  ⚠️  Failed to get trace from PostgreSQL: {e}")
            return None
    
    def list_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent traces from PostgreSQL."""
        if not self._connected:
            return []
        
        try:
            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT data FROM {self._schema}.traces 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (limit,))
                
                return [row["data"] for row in cur.fetchall()]
        except Exception as e:
            print(f"  ⚠️  Failed to list traces from PostgreSQL: {e}")
            return []
    
    def get_runs_by_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get all runs for a trace from PostgreSQL."""
        if not self._connected:
            return []
        
        try:
            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT data FROM {self._schema}.runs 
                    WHERE trace_id = %s 
                    ORDER BY created_at ASC
                """, (trace_id,))
                
                return [row["data"] for row in cur.fetchall()]
        except Exception as e:
            print(f"  ⚠️  Failed to get runs from PostgreSQL: {e}")
            return []
    
    def is_connected(self) -> bool:
        """Check PostgreSQL connection status."""
        return self._connected
    
    def close(self):
        """Close PostgreSQL connection."""
        if self._conn:
            self._conn.close()
            self._connected = False

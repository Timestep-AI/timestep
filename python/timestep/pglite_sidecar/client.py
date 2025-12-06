"""PGlite sidecar client for managing a long-lived Node.js process."""

import os
import json
import asyncio
import shutil
from pathlib import Path
from typing import Optional, Any, Dict
import subprocess


class PGliteSidecarClient:
    """Client for communicating with a PGlite sidecar process via JSON-RPC."""
    
    def __init__(self, pglite_path: str):
        """
        Initialize the sidecar client.
        
        Args:
            pglite_path: Path to the PGlite data directory
        """
        self.pglite_path = Path(pglite_path)
        self.pglite_path.mkdir(parents=True, exist_ok=True)
        self._process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._closed = False
        self._ready_event = asyncio.Event()
        
    def _find_sidecar_script(self) -> Path:
        """Find the sidecar.js script path."""
        # Try to find sidecar.js relative to this module
        script_path = Path(__file__).parent / "sidecar.js"
        if script_path.exists():
            return script_path
        
        # Try to find in package data
        try:
            import timestep
            package_path = Path(timestep.__file__).parent
            script_path = package_path / "pglite_sidecar" / "sidecar.js"
            if script_path.exists():
                return script_path
        except Exception:
            pass
        
        raise RuntimeError(
            "Could not find sidecar.js. "
            "Make sure the PGlite sidecar script is installed."
        )
    
    def _find_node(self) -> str:
        """Find Node.js executable."""
        node_path = shutil.which('node')
        if not node_path:
            raise RuntimeError(
                "Node.js is required for PGLite support. "
                "Install Node.js from https://nodejs.org/ or use PostgreSQL via TIMESTEP_DB_URL"
            )
        return node_path
    
    def _find_pglite_module(self) -> Optional[str]:
        """Find the @electric-sql/pglite module path."""
        # Try to find node_modules in common locations
        possible_paths = [
            Path.cwd() / "node_modules" / "@electric-sql" / "pglite",
            Path(__file__).parent.parent.parent / "node_modules" / "@electric-sql" / "pglite",
            Path.home() / ".npm" / "global" / "node_modules" / "@electric-sql" / "pglite",
        ]
        
        # Also check if pglite is installed globally via npm
        try:
            result = subprocess.run(
                ["npm", "root", "-g"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                global_node_modules = Path(result.stdout.strip())
                possible_paths.append(global_node_modules / "@electric-sql" / "pglite")
        except Exception:
            pass
        
        for path in possible_paths:
            if path.exists() and (path / "package.json").exists():
                return str(path)
        
        return None
    
    async def start(self) -> None:
        """Start the sidecar process."""
        if self._process and self._process.returncode is None:
            return  # Already running
        
        if self._closed:
            raise RuntimeError("Sidecar client has been closed")
        
        node_path = self._find_node()
        script_path = self._find_sidecar_script()
        
        # Check if @electric-sql/pglite is available
        pglite_module = self._find_pglite_module()
        if not pglite_module:
            # Try to require globally - will fail at runtime if not installed
            pass
        
        # Start the sidecar process
        env = os.environ.copy()
        env['PGLITE_PATH'] = str(self.pglite_path.resolve())
        
        # Set NODE_PATH to include global node_modules so sidecar can find @electric-sql/pglite
        try:
            result = subprocess.run(
                ["npm", "root", "-g"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                global_node_modules = result.stdout.strip()
                if global_node_modules:
                    # Append to existing NODE_PATH if set, otherwise set it
                    existing_node_path = env.get('NODE_PATH', '')
                    if existing_node_path:
                        env['NODE_PATH'] = f"{existing_node_path}:{global_node_modules}"
                    else:
                        env['NODE_PATH'] = global_node_modules
        except Exception:
            # If we can't find global node_modules, continue anyway
            # The sidecar will try to find pglite in other ways
            pass
        
        self._process = await asyncio.create_subprocess_exec(
            node_path,
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(Path(__file__).parent.parent.parent)
        )
        
        # Start reading stdout and stderr
        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        
        # Wait for ready notification (with timeout)
        try:
            await asyncio.wait_for(self._wait_for_ready(), timeout=30.0)
        except asyncio.TimeoutError:
            await self.stop()
            raise RuntimeError("Sidecar failed to initialize within 30 seconds")
    
    async def _wait_for_ready(self) -> None:
        """Wait for the sidecar to send ready notification."""
        # Wait for the ready event (with timeout)
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=25.0)
        except asyncio.TimeoutError:
            raise RuntimeError("Sidecar ready notification timeout")
        
        # Give it a moment to fully initialize, then ping to verify
        await asyncio.sleep(0.2)
        try:
            await asyncio.wait_for(self._send_request("ping", {}), timeout=5.0)
        except Exception as e:
            raise RuntimeError(f"Sidecar ping failed: {e}")
    
    async def _read_stdout(self) -> None:
        """Read responses from stdout."""
        if not self._process or not self._process.stdout:
            return
        
        buffer = ''
        while True:
            try:
                chunk = await self._process.stdout.read(4096)
                if not chunk:
                    break
                
                buffer += chunk.decode('utf-8', errors='replace')
                lines = buffer.split('\n')
                buffer = lines.pop() if lines else ''
                
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        response = json.loads(line)
                        await self._handle_response(response)
                    except json.JSONDecodeError:
                        # Ignore invalid JSON
                        pass
            except Exception as e:
                if not self._closed:
                    # Handle error
                    break
    
    async def _read_stderr(self) -> None:
        """Read notifications from stderr."""
        if not self._process or not self._process.stderr:
            return
        
        buffer = ''
        while True:
            try:
                chunk = await self._process.stderr.read(4096)
                if not chunk:
                    break
                
                buffer += chunk.decode('utf-8', errors='replace')
                lines = buffer.split('\n')
                buffer = lines.pop() if lines else ''
                
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        notification = json.loads(line)
                        # Handle notifications (like "ready")
                        if notification.get('method') == 'ready':
                            # Sidecar is ready - set the event
                            self._ready_event.set()
                    except json.JSONDecodeError:
                        # Ignore invalid JSON
                        pass
            except Exception as e:
                if not self._closed:
                    # Handle error
                    break
    
    async def _handle_response(self, response: Dict[str, Any]) -> None:
        """Handle a JSON-RPC response."""
        response_id = response.get('id')
        if response_id is None:
            return  # Notification, not a response
        
        future = self._pending_requests.pop(response_id, None)
        if future:
            if 'error' in response:
                future.set_exception(RuntimeError(
                    f"PGLite error: {response['error'].get('message', 'Unknown error')}"
                ))
            else:
                future.set_result(response.get('result'))
    
    async def _send_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Send a JSON-RPC request and wait for response."""
        if self._closed:
            raise RuntimeError("Sidecar client has been closed")
        
        if not self._process or self._process.returncode is not None:
            await self.start()
        
        self._request_id += 1
        request_id = self._request_id
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }
        
        future = asyncio.Future()
        self._pending_requests[request_id] = future
        
        try:
            if not self._process or not self._process.stdin:
                raise RuntimeError("Sidecar process not available")
            
            request_json = json.dumps(request) + '\n'
            self._process.stdin.write(request_json.encode('utf-8'))
            await self._process.stdin.drain()
            
            # Wait for response with timeout
            try:
                result = await asyncio.wait_for(future, timeout=30.0)
                return result
            except asyncio.TimeoutError:
                self._pending_requests.pop(request_id, None)
                raise RuntimeError(f"Sidecar request timeout: {method}")
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            raise
    
    async def query(self, sql: str, params: Optional[list] = None) -> Dict[str, Any]:
        """Execute a query and return results."""
        result = await self._send_request("query", {
            "sql": sql,
            "params": params or []
        })
        return result
    
    async def execute(self, sql: str, params: Optional[list] = None) -> None:
        """Execute a query without returning results."""
        await self._send_request("execute", {
            "sql": sql,
            "params": params or []
        })
    
    async def fetch(self, sql: str, params: Optional[list] = None) -> list:
        """Fetch rows from a query."""
        result = await self.query(sql, params)
        return result.get('rows', [])
    
    async def fetchrow(self, sql: str, params: Optional[list] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row from a query."""
        rows = await self.fetch(sql, params)
        return rows[0] if rows else None
    
    async def fetchval(self, sql: str, params: Optional[list] = None) -> Any:
        """Fetch a single value from a query."""
        row = await self.fetchrow(sql, params)
        if row:
            return list(row.values())[0] if row else None
        return None
    
    async def stop(self) -> None:
        """Stop the sidecar process."""
        self._closed = True
        
        # Cancel reading tasks
        if self._stdout_task:
            self._stdout_task.cancel()
            try:
                await self._stdout_task
            except asyncio.CancelledError:
                pass
        
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
        
        # Send close request if process is still running
        if self._process and self._process.returncode is None:
            try:
                await self._send_request("close", {})
            except Exception:
                pass
        
        # Terminate process if still running
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except Exception:
                pass
        
        self._process = None
        self._pending_requests.clear()
    
    async def close(self) -> None:
        """Close the sidecar (alias for stop)."""
        await self.stop()


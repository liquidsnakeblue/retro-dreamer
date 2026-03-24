"""WebSocket manager for real-time training metrics streaming."""

import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        if self._loop is None:
            self._loop = asyncio.get_event_loop()

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, data: dict):
        """Broadcast to all connected clients."""
        message = json.dumps(data)
        disconnected = set()
        for ws in self.active_connections:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.add(ws)
        self.active_connections -= disconnected

    def broadcast_sync(self, data: dict):
        """Thread-safe broadcast (call from training thread)."""
        if not self.active_connections or not self._loop:
            return
        message = json.dumps(data)
        for ws in list(self.active_connections):
            try:
                asyncio.run_coroutine_threadsafe(
                    ws.send_text(message), self._loop
                )
            except Exception:
                self.active_connections.discard(ws)

    @property
    def client_count(self) -> int:
        return len(self.active_connections)

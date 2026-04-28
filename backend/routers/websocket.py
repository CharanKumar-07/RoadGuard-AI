# backend/routers/websocket.py
"""WebSocket router for real-time incident broadcasting."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.deps import manager

logger    = logging.getLogger(__name__)
router    = APIRouter(tags=["websocket"])


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Clients connect here to receive JSON push notifications for new incidents.

    Message format:
        {"type": "new_incident", "incident": {...}}
    """
    await manager.connect(websocket)
    logger.info("WebSocket client connected (total: %d)", len(manager.active_connections))
    try:
        while True:
            # Keep connection alive; optionally handle ping/command messages
            data = await websocket.receive_text()
            if data.strip().lower() == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected (total: %d)", len(manager.active_connections))

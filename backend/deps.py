# backend/deps.py
"""
Shared FastAPI dependencies and singletons.

Importing from here avoids circular imports between routers.
"""

from __future__ import annotations

import os
import logging
from typing import Generator

from fastapi import WebSocket
from sqlalchemy.orm import Session

from backend.database.models import Incident

logger = logging.getLogger(__name__)

# ── Evidence directory ────────────────────────────────────────────────────
EVIDENCE_DIR = os.getenv("EVIDENCE_DIR", "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# ── Database session factory (set by main.py) ────────────────────────────
_SessionLocal = None  # type: ignore[assignment]


def init_session_factory(session_local) -> None:
    global _SessionLocal
    _SessionLocal = session_local


def get_db() -> Generator[Session, None, None]:
    if _SessionLocal is None:
        raise RuntimeError("Database not initialised. Call init_session_factory() first.")
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Alert service singleton ──────────────────────────────────────────────
_alert_service = None


def get_alert_service():
    global _alert_service
    if _alert_service is None:
        from backend.alerts.service import AlertService
        _alert_service = AlertService()
    return _alert_service


# ── WebSocket connection manager ─────────────────────────────────────────

class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        dead: list[WebSocket] = []
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.disconnect(conn)


manager = ConnectionManager()


async def broadcast_incident(incident: Incident) -> None:
    """Broadcast a newly created incident to connected WebSocket clients."""
    payload = {
        "type": "new_incident",
        "incident": {
            "id":            incident.id,
            "timestamp":     incident.timestamp.isoformat() if incident.timestamp else None,
            "incident_type": incident.incident_type,
            "camera_id":     incident.camera_id,
            "license_plate": incident.license_plate,
            "status":        incident.status,
            "evidence_image": incident.evidence_image_path,
        },
    }
    await manager.broadcast(payload)

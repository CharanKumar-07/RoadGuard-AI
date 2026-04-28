# backend/routers/incidents.py
"""
Incidents router – CRUD endpoints + stats.
"""

from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.database import crud
from backend.database.models import Base
from backend.deps import get_db, get_alert_service, broadcast_incident, EVIDENCE_DIR

router = APIRouter(prefix="/incidents", tags=["incidents"])


# ── Helper ────────────────────────────────────────────────────────────────

def _save_evidence(file: UploadFile) -> str:
    ext      = (file.filename or "image.jpg").rsplit(".", 1)[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    path     = os.path.join(EVIDENCE_DIR, filename)
    with open(path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    # Return relative path (just filename) so /evidence/{filename} serves it
    return filename


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/", status_code=201)
async def create_incident(
    incident_type: str              = Form(...),
    camera_id:     str              = Form(...),
    license_plate: Optional[str]    = Form(None),
    timestamp:     Optional[str]    = Form(None),
    file:          Optional[UploadFile] = File(None),
    db:            Session          = Depends(get_db),
    alert_svc                       = Depends(get_alert_service),
):
    # Parse timestamp
    ts = None
    if timestamp:
        try:
            ts = datetime.fromisoformat(timestamp)
        except ValueError:
            ts = None

    # Save evidence image if provided
    evidence_path = None
    if file and file.filename:
        evidence_path = _save_evidence(file)

    # Persist
    incident = crud.create_incident(
        db,
        incident_type       = incident_type,
        camera_id           = camera_id,
        license_plate       = license_plate,
        evidence_image_path = evidence_path,
        timestamp           = ts,
    )

    # Broadcast via WebSocket
    await broadcast_incident(incident)

    # Send alerts for accidents and hit-and-run
    if incident_type in ("accident", "hit_and_run"):
        dashboard_url = os.getenv("DASHBOARD_URL", "http://localhost:8501")
        owner         = None
        if license_plate:
            owner = crud.get_owner_by_plate(db, license_plate)

        alert_logs = alert_svc.notify_accident(
            incident_id   = incident.id,
            incident_type = incident_type,
            camera_id     = camera_id,
            timestamp     = (ts or datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S UTC"),
            plate         = license_plate or "",
            dashboard_url = dashboard_url,
            owner_phone   = owner.phone  if owner else None,
            owner_email   = owner.email  if owner else None,
            owner_name    = owner.owner_name if owner else None,
        )

        for log in alert_logs:
            crud.create_alert(
                db,
                incident_id     = incident.id,
                sent_to         = log["sent_to"],
                channel         = log["channel"],
                status          = log["status"],
                message_preview = log.get("message_preview"),
            )

    return {"id": incident.id, "status": incident.status}


@router.get("/")
def list_incidents(
    incident_type: Optional[str]  = None,
    status:        Optional[str]  = None,
    camera_id:     Optional[str]  = None,
    date_from:     Optional[str]  = None,
    date_to:       Optional[str]  = None,
    skip:          int            = 0,
    limit:         int            = 100,
    db:            Session        = Depends(get_db),
):
    def _parse_dt(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None

    incidents = crud.get_incidents(
        db,
        incident_type = incident_type,
        status        = status,
        camera_id     = camera_id,
        date_from     = _parse_dt(date_from),
        date_to       = _parse_dt(date_to),
        skip          = skip,
        limit         = limit,
    )
    return [
        {
            "id":            inc.id,
            "timestamp":     inc.timestamp.isoformat() if inc.timestamp else None,
            "camera_id":     inc.camera_id,
            "incident_type": inc.incident_type,
            "license_plate": inc.license_plate,
            "status":        inc.status,
            "evidence_image": inc.evidence_image_path,
        }
        for inc in incidents
    ]


@router.get("/{incident_id}")
def get_incident(incident_id: int, db: Session = Depends(get_db)):
    inc = crud.get_incident(db, incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")

    owner = None
    if inc.license_plate:
        owner_obj = crud.get_owner_by_plate(db, inc.license_plate)
        if owner_obj:
            owner = {
                "name":    owner_obj.owner_name,
                "phone":   owner_obj.phone,
                "email":   owner_obj.email,
                "vehicle": f"{owner_obj.vehicle_color} {owner_obj.vehicle_make} {owner_obj.vehicle_model}".strip(),
            }

    return {
        "id":              inc.id,
        "timestamp":       inc.timestamp.isoformat() if inc.timestamp else None,
        "camera_id":       inc.camera_id,
        "incident_type":   inc.incident_type,
        "license_plate":   inc.license_plate,
        "status":          inc.status,
        "evidence_image":  inc.evidence_image_path,
        "evidence_video":  inc.evidence_video_path,
        "alerts":          [
            {"sent_to": a.sent_to, "channel": a.channel, "status": a.status}
            for a in inc.alerts
        ],
        "owner":           owner,
    }


@router.put("/{incident_id}/status")
def update_status(
    incident_id: int,
    status:      str,
    db:          Session = Depends(get_db),
):
    allowed = {"pending", "investigating", "resolved"}
    if status not in allowed:
        raise HTTPException(status_code=400, detail=f"Status must be one of {allowed}")
    inc = crud.update_incident_status(db, incident_id, status)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"id": inc.id, "status": inc.status}


@router.delete("/{incident_id}", status_code=200)
def delete_incident(incident_id: int, db: Session = Depends(get_db)):
    ok = crud.delete_incident(db, incident_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"deleted": True, "id": incident_id}


@router.get("/stats/summary")
def stats(db: Session = Depends(get_db)):
    return crud.get_stats(db)


@router.get("/health")
def health():
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════
# Detected Plates
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/{incident_id}/plates")
def get_incident_plates(incident_id: int, db: Session = Depends(get_db)):
    """Return all plates detected for a specific incident."""
    plates = crud.get_plates_by_incident(db, incident_id)
    return [
        {
            "id":                  p.id,
            "timestamp":           p.timestamp.isoformat() if p.timestamp else None,
            "license_plate":       p.license_plate,
            "confidence":          p.confidence,
            "plate_image":         p.plate_image_path,
            "enhanced_image":      p.enhanced_image_path,
            "is_deblurred":        bool(p.is_deblurred),
            "vehicle_track_id":    p.vehicle_track_id,
        }
        for p in plates
    ]


# Separate router for /plates/ endpoints
from fastapi import APIRouter as _APIRouter
plates_router = _APIRouter(prefix="/plates", tags=["plates"])


@plates_router.get("/")
def list_plates(
    skip:          int            = 0,
    limit:         int            = 100,
    license_plate: Optional[str]  = None,
    db:            Session        = Depends(get_db),
):
    """List all detected plates with optional search filter."""
    plates = crud.get_all_plates(db, skip=skip, limit=limit, license_plate=license_plate)
    return [
        {
            "id":                  p.id,
            "timestamp":           p.timestamp.isoformat() if p.timestamp else None,
            "camera_id":           p.camera_id,
            "license_plate":       p.license_plate,
            "confidence":          p.confidence,
            "plate_image":         p.plate_image_path,
            "enhanced_image":      p.enhanced_image_path,
            "is_deblurred":        bool(p.is_deblurred),
            "incident_id":         p.incident_id,
            "vehicle_track_id":    p.vehicle_track_id,
        }
        for p in plates
    ]


@plates_router.get("/stats")
def plate_stats(db: Session = Depends(get_db)):
    """Return plate detection statistics."""
    return crud.get_plate_stats(db)


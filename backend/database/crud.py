# backend/database/crud.py
"""
CRUD operations for all database models.

All functions accept a SQLAlchemy Session as their first argument.
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from .models import Incident, Alert, VehicleOwner, DetectedPlate


# ═══════════════════════════════════════════════════════════════════════════
# Incident
# ═══════════════════════════════════════════════════════════════════════════

def create_incident(
    db:                   Session,
    incident_type:        str,
    camera_id:            str,
    license_plate:        Optional[str]  = None,
    evidence_image_path:  Optional[str]  = None,
    evidence_video_path:  Optional[str]  = None,
    timestamp:            Optional[datetime] = None,
) -> Incident:
    incident = Incident(
        incident_type       = incident_type,
        camera_id           = camera_id,
        license_plate       = license_plate or None,
        evidence_image_path = evidence_image_path,
        evidence_video_path = evidence_video_path,
        timestamp           = timestamp or datetime.utcnow(),
        status              = "pending",
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return incident


def get_incidents(
    db:            Session,
    incident_type: Optional[str]      = None,
    status:        Optional[str]      = None,
    camera_id:     Optional[str]      = None,
    date_from:     Optional[datetime] = None,
    date_to:       Optional[datetime] = None,
    skip:          int = 0,
    limit:         int = 100,
) -> list[Incident]:
    q = db.query(Incident)
    if incident_type:
        q = q.filter(Incident.incident_type == incident_type)
    if status:
        q = q.filter(Incident.status == status)
    if camera_id:
        q = q.filter(Incident.camera_id == camera_id)
    if date_from:
        q = q.filter(Incident.timestamp >= date_from)
    if date_to:
        q = q.filter(Incident.timestamp <= date_to)
    return q.order_by(Incident.timestamp.desc()).offset(skip).limit(limit).all()


def get_incident(db: Session, incident_id: int) -> Optional[Incident]:
    return db.query(Incident).filter(Incident.id == incident_id).first()


def update_incident_status(db: Session, incident_id: int, status: str) -> Optional[Incident]:
    incident = get_incident(db, incident_id)
    if not incident:
        return None
    incident.status = status
    db.commit()
    db.refresh(incident)
    return incident


def delete_incident(db: Session, incident_id: int) -> bool:
    incident = get_incident(db, incident_id)
    if not incident:
        return False
    db.delete(incident)
    db.commit()
    return True


def count_incidents(db: Session) -> int:
    return db.query(func.count(Incident.id)).scalar()


# ═══════════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════════

def get_stats(db: Session) -> dict:
    """
    Returns:
        total        – total incident count
        by_type      – dict incident_type → count
        by_status    – dict status → count
        daily        – list of {date, count} for last 30 days
    """
    total = db.query(func.count(Incident.id)).scalar()

    by_type = {
        row[0]: row[1]
        for row in db.query(Incident.incident_type, func.count(Incident.id))
                      .group_by(Incident.incident_type).all()
    }

    by_status = {
        row[0]: row[1]
        for row in db.query(Incident.status, func.count(Incident.id))
                      .group_by(Incident.status).all()
    }

    # Daily counts (group by date string — works for SQLite and PostgreSQL)
    daily_rows = (
        db.query(
            func.strftime("%Y-%m-%d", Incident.timestamp).label("day"),
            func.count(Incident.id).label("count"),
        )
        .group_by("day")
        .order_by("day")
        .limit(30)
        .all()
    )
    daily = [{"date": row.day, "count": row.count} for row in daily_rows]

    return {
        "total":     total,
        "by_type":   by_type,
        "by_status": by_status,
        "daily":     daily,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Alert
# ═══════════════════════════════════════════════════════════════════════════

def create_alert(
    db:              Session,
    incident_id:     int,
    sent_to:         str,
    channel:         str,
    status:          str = "sent",
    message_preview: Optional[str] = None,
) -> Alert:
    alert = Alert(
        incident_id     = incident_id,
        sent_to         = sent_to,
        channel         = channel,
        status          = status,
        message_preview = message_preview,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


# ═══════════════════════════════════════════════════════════════════════════
# Vehicle Owner
# ═══════════════════════════════════════════════════════════════════════════

def get_owner_by_plate(db: Session, license_plate: str) -> Optional[VehicleOwner]:
    return (
        db.query(VehicleOwner)
          .filter(VehicleOwner.license_plate == license_plate.upper().strip())
          .first()
    )


def create_owner(
    db:            Session,
    license_plate: str,
    owner_name:    str,
    phone:         str,
    email:         str,
    address:       str  = "",
    vehicle_make:  str  = "",
    vehicle_model: str  = "",
    vehicle_color: str  = "",
) -> VehicleOwner:
    owner = VehicleOwner(
        license_plate = license_plate.upper().strip(),
        owner_name    = owner_name,
        phone         = phone,
        email         = email,
        address       = address,
        vehicle_make  = vehicle_make,
        vehicle_model = vehicle_model,
        vehicle_color = vehicle_color,
    )
    db.add(owner)
    db.commit()
    db.refresh(owner)
    return owner


# ═══════════════════════════════════════════════════════════════════════════
# Detected Plates
# ═══════════════════════════════════════════════════════════════════════════

def create_detected_plate(
    db:                   Session,
    camera_id:            str,
    license_plate:        str,
    confidence:           float = 0.0,
    plate_image_path:     Optional[str] = None,
    enhanced_image_path:  Optional[str] = None,
    incident_id:          Optional[int] = None,
    vehicle_track_id:     Optional[int] = None,
    is_deblurred:         bool = False,
) -> "DetectedPlate":
    plate = DetectedPlate(
        camera_id           = camera_id,
        license_plate       = license_plate.upper().strip(),
        confidence          = confidence,
        plate_image_path    = plate_image_path,
        enhanced_image_path = enhanced_image_path,
        incident_id         = incident_id,
        vehicle_track_id    = vehicle_track_id,
        is_deblurred        = 1 if is_deblurred else 0,
    )
    db.add(plate)
    db.commit()
    db.refresh(plate)
    return plate


def get_plates_by_incident(db: Session, incident_id: int) -> list:
    return (
        db.query(DetectedPlate)
          .filter(DetectedPlate.incident_id == incident_id)
          .order_by(DetectedPlate.timestamp.desc())
          .all()
    )


def get_all_plates(
    db:    Session,
    skip:  int = 0,
    limit: int = 100,
    license_plate: Optional[str] = None,
) -> list:
    q = db.query(DetectedPlate)
    if license_plate:
        q = q.filter(DetectedPlate.license_plate.contains(license_plate.upper()))
    return q.order_by(DetectedPlate.timestamp.desc()).offset(skip).limit(limit).all()


def get_plate_stats(db: Session) -> dict:
    """Return plate detection statistics."""
    total = db.query(func.count(DetectedPlate.id)).scalar()
    unique = db.query(func.count(func.distinct(DetectedPlate.license_plate))).scalar()
    deblurred = db.query(func.count(DetectedPlate.id)).filter(DetectedPlate.is_deblurred == 1).scalar()
    with_incident = db.query(func.count(DetectedPlate.id)).filter(DetectedPlate.incident_id.isnot(None)).scalar()
    return {
        "total_detections": total,
        "unique_plates": unique,
        "deblurred_count": deblurred,
        "linked_to_incident": with_incident,
    }


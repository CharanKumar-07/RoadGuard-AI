# backend/database/models.py
"""SQLAlchemy ORM models for RoadGuard AI."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, DateTime, Float,
    ForeignKey, Text, Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Incident(Base):
    __tablename__ = "incidents"

    id                  = Column(Integer, primary_key=True, index=True)
    timestamp           = Column(DateTime, default=datetime.utcnow, index=True)
    camera_id           = Column(String(50), nullable=False, index=True)
    incident_type       = Column(String(50), nullable=False, index=True)
    # accident | hit_and_run | wrong_way | speed | red_light
    license_plate       = Column(String(30), nullable=True, index=True)
    evidence_image_path = Column(String(300), nullable=True)
    evidence_video_path = Column(String(300), nullable=True)
    status              = Column(String(20), default="pending", index=True)
    # pending | investigating | resolved

    alerts = relationship("Alert", back_populates="incident", cascade="all, delete-orphan")


class Alert(Base):
    __tablename__ = "alerts"

    id              = Column(Integer, primary_key=True, index=True)
    incident_id     = Column(Integer, ForeignKey("incidents.id"), nullable=False)
    sent_to         = Column(String(150))      # phone number or email address
    channel         = Column(String(20))       # sms | email
    timestamp       = Column(DateTime, default=datetime.utcnow)
    status          = Column(String(20), default="sent")  # sent | delivered | failed
    message_preview = Column(Text, nullable=True)

    incident = relationship("Incident", back_populates="alerts")


class VehicleOwner(Base):
    __tablename__ = "vehicle_owners"

    id            = Column(Integer, primary_key=True, index=True)
    license_plate = Column(String(30), unique=True, nullable=False, index=True)
    owner_name    = Column(String(150))
    phone         = Column(String(25))
    email         = Column(String(150))
    address       = Column(Text, nullable=True)
    vehicle_make  = Column(String(60), nullable=True)
    vehicle_model = Column(String(60), nullable=True)
    vehicle_color = Column(String(30), nullable=True)


class DetectedPlate(Base):
    """
    Every license plate detection is stored here for forensic audit.
    Linked to an incident when the plate is associated with one.
    """
    __tablename__ = "detected_plates"

    id                  = Column(Integer, primary_key=True, index=True)
    timestamp           = Column(DateTime, default=datetime.utcnow, index=True)
    camera_id           = Column(String(50), nullable=False)
    license_plate       = Column(String(30), nullable=False, index=True)
    confidence          = Column(Float, default=0.0)
    plate_image_path    = Column(String(300), nullable=True)
    enhanced_image_path = Column(String(300), nullable=True)
    incident_id         = Column(Integer, ForeignKey("incidents.id"), nullable=True, index=True)
    vehicle_track_id    = Column(Integer, nullable=True)
    is_deblurred        = Column(Integer, default=0)  # 0=no, 1=yes

    incident = relationship("Incident", backref="detected_plates")
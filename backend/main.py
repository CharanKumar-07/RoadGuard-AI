# backend/main.py
"""
RoadGuard AI – FastAPI Backend
==============================

Start:  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base
from backend.deps             import init_session_factory, EVIDENCE_DIR
from backend.routers          import incidents as incidents_router
from backend.routers          import websocket as websocket_router
from backend.routers          import video_processor as video_router

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")

# ── Database ──────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./roadguard.db",         # SQLite for development (zero-config)
)

# PostgreSQL note: switch to postgresql://user:pass@postgres/roadguard in .env
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine       = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "RoadGuard AI – API",
    description = "Intelligent traffic surveillance backend.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Static evidence files ─────────────────────────────────────────────────
os.makedirs(EVIDENCE_DIR, exist_ok=True)
app.mount("/evidence", StaticFiles(directory=EVIDENCE_DIR), name="evidence")

# ── Routers ───────────────────────────────────────────────────────────────
app.include_router(incidents_router.router)
app.include_router(websocket_router.router)
app.include_router(video_router.router)

# Plates sub-router (defined in incidents module)
from backend.routers.incidents import plates_router
app.include_router(plates_router)


# ── Root ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "RoadGuard AI API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats_redirect():
    """Convenience alias for /incidents/stats/summary."""
    from backend.deps import get_db
    from backend.database import crud
    db = SessionLocal()
    try:
        return crud.get_stats(db)
    finally:
        db.close()


# ── Startup ───────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Creating database tables …")
    Base.metadata.create_all(bind=engine)
    init_session_factory(SessionLocal)

    # Seed mock vehicle owners the first time
    db = SessionLocal()
    try:
        _seed_mock_data(db)
    finally:
        db.close()

    logger.info("RoadGuard AI backend started. DATABASE_URL=%s", DATABASE_URL)


def _seed_mock_data(db) -> None:
    """Populate VehicleOwner table with demo data if empty."""
    from backend.database.models import VehicleOwner
    from backend.database.crud   import create_owner

    if db.query(VehicleOwner).count() > 0:
        return  # already seeded

    mock_owners = [
        dict(license_plate="MH12AB1234", owner_name="Arjun Patel",     phone="+911234567890", email="arjun@example.com",  vehicle_make="Honda",       vehicle_model="City",     vehicle_color="White"),
        dict(license_plate="DL09CD5678", owner_name="Priya Sharma",    phone="+911234567891", email="priya@example.com",  vehicle_make="Maruti",      vehicle_model="Swift",    vehicle_color="Silver"),
        dict(license_plate="KA01EF9012", owner_name="Rahul Kumar",     phone="+911234567892", email="rahul@example.com",  vehicle_make="Hyundai",     vehicle_model="i20",      vehicle_color="Red"),
        dict(license_plate="TN07GH3456", owner_name="Deepa Nair",      phone="+911234567893", email="deepa@example.com",  vehicle_make="Toyota",      vehicle_model="Innova",   vehicle_color="Grey"),
        dict(license_plate="UP32IJ7890", owner_name="Vikram Singh",    phone="+911234567894", email="vikram@example.com", vehicle_make="Tata",        vehicle_model="Nexon",    vehicle_color="Blue"),
        dict(license_plate="GJ01KL2345", owner_name="Meera Desai",     phone="+911234567895", email="meera@example.com",  vehicle_make="Mahindra",    vehicle_model="Scorpio",  vehicle_color="Black"),
        dict(license_plate="RJ14MN6789", owner_name="Suresh Yadav",    phone="+911234567896", email="suresh@example.com", vehicle_make="Ford",        vehicle_model="EcoSport", vehicle_color="Orange"),
        dict(license_plate="WB23OP0123", owner_name="Anjali Banerjee", phone="+911234567897", email="anjali@example.com", vehicle_make="Volkswagen",  vehicle_model="Polo",     vehicle_color="Yellow"),
    ]
    for data in mock_owners:
        try:
            create_owner(db, **data)
        except Exception:
            db.rollback()
    logger.info("Seeded %d mock vehicle owners.", len(mock_owners))
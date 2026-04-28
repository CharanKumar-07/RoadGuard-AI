#!/usr/bin/env python3
"""
scripts/seed_db.py
==================
Populate the database with mock vehicle owners and sample incidents.

Usage:
    python scripts/seed_db.py [--backend http://localhost:8000]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import os
from datetime import datetime, timedelta

import requests

TYPES    = ["accident", "hit_and_run", "wrong_way", "speed"]
STATUSES = ["pending", "investigating", "resolved"]
CAMERAS  = ["cam_01", "cam_02", "cam_03"]
PLATES   = [
    "MH12AB1234", "DL09CD5678", "KA01EF9012",
    "TN07GH3456", "UP32IJ7890", "GJ01KL2345",
    "RJ14MN6789", "WB23OP0123", None, None,   # some with no plate
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--count",   type=int, default=20, help="Number of sample incidents to create")
    args = parser.parse_args()

    base = args.backend.rstrip("/")

    # Health check
    try:
        hc = requests.get(f"{base}/health", timeout=5)
        hc.raise_for_status()
    except Exception as exc:
        print(f"[ERROR] Cannot reach backend at {base}: {exc}")
        sys.exit(1)

    print(f"Connected to backend at {base}")

    # Create a tiny 64×64 red test image (JPEG bytes) as dummy evidence
    import io
    try:
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (64, 64), color=(200, 50, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()
    except ImportError:
        # Fallback: minimal valid JPEG
        img_bytes = bytes([0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,
                           0x01,0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xD9])

    created = 0
    for i in range(args.count):
        ts      = (datetime.utcnow() - timedelta(hours=random.randint(0, 72))).isoformat()
        itype   = random.choice(TYPES)
        camera  = random.choice(CAMERAS)
        plate   = random.choice(PLATES)
        status  = random.choice(STATUSES)

        try:
            resp = requests.post(
                f"{base}/incidents/",
                data={
                    "incident_type": itype,
                    "camera_id":     camera,
                    "license_plate": plate or "",
                    "timestamp":     ts,
                },
                files={"file": ("dummy.jpg", io.BytesIO(img_bytes), "image/jpeg")},
                timeout=10,
            )
            if resp.ok:
                inc_id = resp.json().get("id")
                # Update to a non-pending status occasionally
                if status != "pending":
                    requests.put(f"{base}/incidents/{inc_id}/status", params={"status": status}, timeout=5)
                created += 1
                print(f"  [{created}/{args.count}] {itype} | {camera} | {plate or '(no plate)'} | {status}")
            else:
                print(f"  [WARN] POST failed: {resp.status_code} {resp.text[:80]}")
        except Exception as exc:
            print(f"  [ERROR] {exc}")

    print(f"\nDone. Created {created} sample incidents.")


if __name__ == "__main__":
    main()

# RoadGuard AI 🚦

**Intelligent Traffic Surveillance Platform** — real-time accident detection, ANPR (Automatic Number Plate Recognition), hit-and-run monitoring, and wrong-way driving alerts, powered by YOLOv8, DeblurGAN-v2, and EasyOCR.

---

## Architecture

```
[Video Streams] ──► [Processor (Python)] ──► [Backend (FastAPI)] ──► [Dashboard (Next.js)]
        │                        │
        ▼                        ▼
      [Evidence Storage]       [SQLite / PostgreSQL]
        │                        │
        ▼                        ▼
      [Alert Service] ◄──── [Incident Data]
```

## Features

| Feature              | Description |
|----------------------|-------------|
| Accident Detection   | YOLOv8 model detects road accidents in real-time |
| ANPR                 | Plate detection → DeblurGAN enhancement → EasyOCR |
| Hit-and-Run          | Flags vehicles that flee the accident scene within 10 s |
| Wrong-Way Driving    | Lane-direction logic with clickable calibration tool |
| Multi-Camera         | Concurrent processing of multiple RTSP / file / webcam streams |
| Dashboard            | Live feed, filterable table, incident detail, analytics charts |
| Alerts               | SMS (Twilio) + HTML email (SMTP) — optional |
| REST API             | FastAPI with OpenAPI docs at `/docs` |
| WebSocket            | Real-time push to connected dashboards |
| Docker               | Single `docker-compose up` deployment |

---

## Quick Start

### Option 1 – Local (without Docker)

**1. Clone & set up environment**
```bash
git clone https://github.com/your-org/RoadGuardAI.git
cd RoadGuardAI
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

**2. Copy config files**
```bash
cp config/cameras.yaml.example    config/cameras.yaml
cp config/calibration.yaml.example config/calibration.yaml
cp .env.example .env
# Edit config/cameras.yaml to point at your video sources
```

**3. Start the backend**
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
# Opens SQLite database at ./roadguard.db — no Postgres needed
```

**4. (Optional) Seed sample data**
```bash
python scripts/seed_db.py --count 30
```

**5. Start the dashboard (Next.js)**
```bash
cd roadguard-next
npm install
npm run dev
# Open http://localhost:3000
```

**6. Start the processor** (in another terminal)
```bash
python -m src.processor --cameras config/cameras.yaml
```

---

### Option 2 – Docker Compose (recommended for production)

```bash
cp .env.example .env
# Edit .env — set Twilio/SMTP credentials and emergency contacts if needed

docker-compose up --build
```

Services:
| Service       | URL                         |
|---------------|----------------------------|
| Backend API   | http://localhost:8000       |
| API Docs      | http://localhost:8000/docs  |
| PostgreSQL    | localhost:5432              |

---

## Configuration

### cameras.yaml
```yaml
cameras:
  - id: cam_01
    source: "0"              # webcam index
    fps: 25
    enabled: true

  - id: cam_02
    source: "rtsp://user:pass@192.168.1.100:554/stream"
    enabled: true

  - id: cam_03
    source: "data/test_video.mp4"
    enabled: true
```

### calibration.yaml
```yaml
calibrations:
  - camera_id: cam_01
    lanes:
      - id: lane_left
        vector: [1, 0]        # allowed direction: left → right
        polygon: [[0,200],[640,200],[640,480],[0,480]]
```

**Interactive calibration tool:**
```bash
python scripts/calibrate_camera.py --source 0 --camera-id cam_01
# Click two points on the frame to define the allowed direction
# Press S to save, R to reset, Q to quit
```

### Environment Variables (.env)
| Variable              | Default                    | Description |
|-----------------------|----------------------------|-------------|
| `DATABASE_URL`        | `sqlite:///./roadguard.db` | DB connection string |
| `BACKEND_URL`         | `http://localhost:8000`    | URL the processor posts to |
| `EVIDENCE_DIR`        | `evidence`                 | Where frame captures are stored |
| `ACCIDENT_CONF`       | `0.50`                     | Accident detection confidence threshold |
| `TWILIO_ACCOUNT_SID`  | *(optional)*               | Twilio account SID for SMS |
| `TWILIO_AUTH_TOKEN`   | *(optional)*               | Twilio auth token |
| `TWILIO_FROM_NUMBER`  | *(optional)*               | Twilio sender number |
| `SMTP_USER`           | *(optional)*               | Gmail / SMTP username |
| `SMTP_PASS`           | *(optional)*               | Gmail app password |
| `EMERGENCY_PHONES`    | *(optional)*               | Comma-separated emergency numbers |
| `EMERGENCY_EMAILS`    | *(optional)*               | Comma-separated emergency emails |

---

## Models

| Model        | File                    | Purpose |
|--------------|-------------------------|---------|
| Accident     | `models/acc_detect.pt`  | YOLOv8 – accident scene detection |
| Vehicle      | `models/yolov8n.pt`     | YOLOv8 COCO – vehicle tracking |
| Plate        | `models/yolo26n.pt`     | YOLOv8 – license plate localisation |
| DeblurGAN-v2 | `models/fpn_inception.h5` | Keras – plate image enhancement |

---

## API Reference

| Method | Endpoint                       | Description |
|--------|--------------------------------|-------------|
| `POST` | `/incidents/`                  | Create an incident (multipart form + image) |
| `GET`  | `/incidents/`                  | List incidents (filter by type, status, date) |
| `GET`  | `/incidents/{id}`              | Full incident detail with owner + alert logs |
| `PUT`  | `/incidents/{id}/status`       | Update status (pending/investigating/resolved) |
| `GET`  | `/incidents/stats/summary`     | Aggregated stats (by type, status, daily) |
| `GET`  | `/health`                      | Health check |
| `WS`   | `/ws`                          | WebSocket for real-time incident push |
| `GET`  | `/evidence/{filename}`         | Static evidence image serving |

Full interactive docs: **http://localhost:8000/docs**

---

## Project Structure

```
RoadGuardAI/
├── models/                      # Trained model files (.pt, .h5)
├── src/
│   ├── processor.py             # Main multi-camera pipeline
│   ├── detection/yolo_detector.py
│   ├── tracking/tracker.py      # IoU multi-object tracker
│   ├── anpr/
│   │   ├── pipeline.py          # end-to-end ANPR
│   │   ├── plate_detector.py
│   │   ├── deblurrer.py         # DeblurGAN-v2 (Keras)
│   │   └── ocr_reader.py        # EasyOCR wrapper
│   ├── violations/
│   │   ├── wrong_way.py
│   │   └── hit_and_run.py
│   └── utils/
│       ├── video_reader.py      # Multi-source reader + auto-reconnect
│       ├── config.py            # YAML config loaders
│       └── calibration.py
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── deps.py                  # Shared DB / alert / WS singletons
│   ├── database/
│   │   ├── models.py            # SQLAlchemy ORM
│   │   └── crud.py
│   ├── routers/
│   │   ├── incidents.py
│   │   └── websocket.py
│   └── alerts/service.py
├── roadguard-next/              # Next.js dashboard
├── config/                      # Camera + calibration YAML examples
├── scripts/
│   ├── calibrate_camera.py      # Interactive calibration
│   └── seed_db.py               # Demo data seeder
├── evidence/                    # Auto-created; stores captured frames
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Cannot open video source` | Check camera index / RTSP URL; try `source: "0"` for webcam |
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow>=2.14` — required for DeblurGAN |
| `easyocr` install fails | Try `pip install easyocr` and verify PyTorch/CUDA compatibility |
| Backend not reachable from processor | Ensure `BACKEND_URL` in `.env` matches the backend host |
| `roadguard.db` locked | Stop all services sharing the SQLite file before restarting |
| Duplicate incidents | Increase `ACCIDENT_CONF` to 0.6–0.7 to reduce false positives |
| No alerts sent | Verify Twilio/SMTP credentials in `.env` and check logs |

---

## License

MIT License — see `LICENSE` for details.

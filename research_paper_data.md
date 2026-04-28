# RoadGuard AI: Intelligent Traffic Surveillance and Forensic Evidence Collection System

This document contains a highly detailed compilation of research data on the **RoadGuard AI** project. You can provide this text directly to another LLM to structure, expand, and format it into a comprehensive 6-page academic research paper.

---

## 1. Abstract and Introduction 
**Objective:** RoadGuard AI is an advanced, real-time traffic surveillance platform designed to automate accident detection, multi-object vehicle tracking, Automatic Number Plate Recognition (ANPR), and traffic violation monitoring (hit-and-run, wrong-way driving).
**Motivation:** Traditional CCTV systems rely on manual monitoring, leading to delayed emergency responses and uncaptured forensic evidence in fast-paced hit-and-run scenarios. The project addresses this via a multi-staged deep learning pipeline.
**Contribution:** A unified framework integrating state-of-the-art object detection (YOLOv8), image enhancement (DeblurGAN-v2), and text extraction (EasyOCR) interconnected with a high-performance ASGI backend (FastAPI) and real-time visualization.

---

## 2. System Architecture
The platform is designed around a decoupled, microservices-style architecture capable of handling multiple asynchronous video streams (RTSP, webcam, local files).
**Components:**
1. **Video Processor Pipeline (Python):** Ingests frames and runs inference using optimized AI models. Utilizes auto-reconnection wrappers for unstable RTSP streams.
2. **Backend Engine (FastAPI):** Exposes REST API endpoints for incident logging and manages automated alerts (Twilio SMS / SMTP Emails). Leverages WebSockets for pushing real-time incident updates to connected clients.
3. **Database Layer:** Uses SQLAlchemy ORM. Supports SQLite for local deployments and PostgreSQL for production environments (`roadguard.db`).
4. **Dashboard:** 
   - *Streamlit Application:* Facilitates raw data filtering, database management, tabular representations, and analytics charts.
   - *Next.js 14 Application (`roadguard-next`):* A premium, Apple-style interactive graphical interface built with Framer motion, utilizing a 120-frame image sequence canvas for dynamic scrollytelling.
5. **Deployment:** Entire architecture is containerized using Docker and easily deployable via `docker-compose up`.

---

## 3. Deep Learning Pipeline and AI Models
The core of RoadGuard AI revolves around multi-layered inference models to ensure fault tolerance.

### 3.1 Object and Accident Detection (YOLOv8)
- **Accident Model (`models/acc_detect.pt`):** A custom-trained YOLOv8 model tailored specifically for detecting chaotic accident scenes (e.g., collisions, wrecks). It operates with an optimal confidence threshold (0.50 to 0.70) to mitigate false positives in dense traffic.
- **Vehicle Tracking (`models/yolov8n.pt`):** Pre-trained YOLOv8 Nano COCO model tracks individual vehicles in the scene.
- **Plate Localization (`models/yolo26n.pt`):** Captures tightly bound bounding boxes specifically identifying license plates in tracked vehicles.

### 3.2 Advanced Multi-Object Tracking (IoU Tracker)
- **Methodology:** Implemented a lightweight, ByteTrack-inspired custom Intersection over Union (IoU) tracker (`src/tracking/tracker.py`). 
- **Mechanism:** Assigns and maintains stable integer IDs to moving objects based on frame-by-frame spatial overlap. Features tracking persistence—keeping the track alive for up to a designated threshold of 'missed frames' before culling, mitigating transient occlusion failures.

### 3.3 Enhanced ANPR (DeblurGAN-v2 & EasyOCR)
- **Image Enhancement:** Frequently, fast-moving vehicles yield blurred frame captures. RoadGuard AI passes the localized plate bbox into heavily optimized **DeblurGAN-v2** (`fpn_inception.h5` in Keras/TensorFlow) which sharpens edge lines and eliminates motion blur prior to OCR.
- **Lazy 3-Pass OCR Extraction (`src/anpr/ocr_reader.py`):** Uses an EasyOCR setup employing a lazy execution strategy to maximize speed. It stops as soon as a format-compliant string is identified.
  1. *Pass 1 (CLAHE Enhancement):* Most reliable under poor lighting; applies Bilateral filtering and Adaptive Histogram Equalization.
  2. *Pass 2 (Raw Crop):* A rapid pass over unmodified/upscaled pixels. Highly efficient for sharp inputs.
  3. *Pass 3 (Binary Thresholding):* Fallback approach utilizing OpenCV `THRESH_BINARY_INV`.
- **Intelligent Correction:** Handles OCR hallucinations dynamically by incorporating character mapping (e.g., confusing `O`/`0`, `I`/`1`) mapped explicitly to international (7-char) and Indian (10-char format: `MH12AB1234`) license plate standards. 
- **Hash Caching Optimization:** Creates rapid MD5 perceptual hashes of downscaled 16x8 crops. Before utilizing GPU/CPU inference, the system checks the memory cache (`max_size: 512`), skipping duplicate frame readings instantly. 

---

## 4. Algorithmic Modules for Traffic Violations

### 4.1 Hit-and-Run Detection
The hit-and-run algorithm (`src/violations/hit_and_run.py`) activates immediately after an accident event triggers. 
1. **Scene Snapshotting:** On an accident trigger, it snapshots the centroid geometry and compiles a list of uniquely ID'd vehicles that have bounding boxes overlapping the accident area (IoU > 0.15) or fall within a configurable pixel radius (default 200px).
2. **Zone Monitoring:** These stored tracker IDs are placed in an observation pool.
3. **Timeout Evaluation:** If an involved tracking ID vanishes from the frame entirely, or progresses geographically past the accident radius before a mandated cooldown timeframe (e.g., 10 seconds), the system permanently flags that vehicle for hit-and-run violations and archives the associated high-resolution evidence frame.

### 4.2 Wrong-Way Driving Constraints
Operates using geometrically calibrated polygons. A calibration config (`calibration.yaml`) defines physical coordinates of traffic lanes and allowed travel vectors.
- Vehicles are continuously tracked via centroid analysis. If the vector difference between historical centroids and active centroid heavily opposes the legal lane vector, a wrong-way anomaly is recorded.

---

## 5. Backend Logic, Security, and Edge Limitations
- **Idempotent Logs:** System ensures duplicated crash logs don't spam databases by leveraging interval cooldowns tied to camera IDs and geographical areas within the frame.
- **Evidence Persistence:** Automatically saves full resolution incident frames locally in `/evidence` whilst only indexing the base paths and associated violation metadata to the SQLite/PostgreSQL layer.
- **Extensibility:** The FastAPI router structure separates dependencies, schemas, and endpoints concisely, preventing monolith overhead. 

---

## 6. Experimental Results and Performance Improvements
- **Speed Benchmarking:** The shift from a naïve OCR approach to the Lazy 3-pass + MD5 Caching system drastically improved processing speed per frame. Caching prevented redundant deep learning inference calls.
- **Accuracy Improvement:** The incorporation of DeblurGAN-v2 directly tackled the largest failure point of standard ANPR arrays (motion blurring), significantly lifting the F1 Score accuracy of extracted strings compared to un-enhanced images.
- **False Positive Reduction:** Limiting the hit-and-run tracking only to explicitly overlapping or physically proximal vehicle tracks (IoU > 0.15) eliminated routine bystanders from triggering false reports.

---

## 7. Future Scope
- Transitioning to 3D spatial trackers relying on homography matrices rather than pixel-space 2D tracking.
- Utilizing edge TPUs (Tensor Processing Units) for strictly distributed local inference prior to centralized cloud aggregating.
- Integration of multi-camera vehicle Re-ID (Re-identification) algorithms to track vehicles sequentially over entirely separate IP cameras.

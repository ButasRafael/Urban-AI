# 🌆 Urban AI: Comprehensive Smart City Issue Detection, Segmentation, and Reporting

Urban AI provides a complete solution for efficiently identifying, segmenting, and managing urban issues such as potholes, graffiti, illegal parking, garbage, cracks, fallen trees, and others. The project uses advanced artificial intelligence techniques, including YOLOv11 for rapid object detection, Grounding Dino—leveraging GPT-4.1’s newly identified detections with their carefully chosen dino prompts produce accurate bounding boxes where GPT alone cannot—GPT-4.1 to validate YOLO detections, identify any missed issues, and generate human-readable descriptions and remediation suggestions, and Segment Anything Model 2 (SAM2) for precise segmentation. Integrated with a powerful backend, intuitive web interface, and user-friendly mobile application, Urban AI empowers citizens to report problems seamlessly and allows authorities to manage urban maintenance effectively.

## 📑 Table of Contents

* [Project Overview](#project-overview)
* [AI Model Development and Training](#ai-model-development-and-training)
* [Backend Development](#backend-development)
  * [Monitoring and Logging](#monitoring-and-logging)
    * [Prometheus & Grafana](#prometheus--grafana)
    * [Loki & Tempo](#loki--tempo)
* [Frontend Applications](#frontend-applications)
* [Chat & RAG Module](#-chat--rag-module)
* [Docker Services Explained](#docker-services-explained)
* [Installation and Usage Guide](#installation-and-usage-guide)
* [Contributing and Development](#contributing-and-development)


## 🌟 Project Overview

Urban AI simplifies the reporting and management of city issues by utilizing AI-driven analyses of user-uploaded photos and videos. Reports are processed automatically, generating accurate classifications and segmentations of urban problems, significantly enhancing response times and management efficiency.

## 🧠 AI Model Development and Training (Expanded)

### 1. Dataset Construction & Label Remapping
- **Raw sources**: We aggregate 30+ public datasets covering 12 classes (potholes, graffiti, garbage, bins, overflows, illegal/empty/legal parking, cracks, manholes, fallen trees).  
- **ID mapping**: A per-dataset `IDMAP` remaps source class IDs → our canonical 0–11 label set, e.g. all “pothole” variants → 0, “graffiti” → 1, etc.  
- **Split management**:  
  - If a dataset lacks a `train` split, we borrow from its `valid`/`test` directories.  
  - Missing `valid`/`test` splits are synthesized by sampling 10 % of the `train` set.  
- **Deduplication**: Images appearing in both `train` and (`valid` / `test`) are removed from `train` to prevent leakage.  
- **YAML manifest**: After copying & remapping labels/images into `urban_yolo_final_all/{images,labels}/{train,valid,test}`, we auto-generate `urban_yolo_final_all.yaml` with `nc=12` and our class names.

### 2. Label Cleaning
- A quick pass over every `.txt` label file in train/valid/test to strip out malformed lines (ensuring exactly 5 tab-delimited fields per bounding box: `class_id x_center y_center width height`).

### 3. Data Augmentation (Albumentations-Based)
We perform **class-aware oversampling** to balance rare classes up to the 90th percentile of per-class image counts:
1. **Count per class** → compute target augmentation count for each label.  
2. **Augmentation pipeline** (`A.Compose`, `bbox_params=yolo`):  
   - **Geometric**: random resized crop, flips, affine, perspective  
   - **Photometric**: contrast, brightness, color jitter, posterize, RGB shift  
   - **Blur/Noise**: Gaussian, motion, glass, shot/salt-pepper, ISO noise  
   - **Weather**: fog, rain, shadows, snow, sun flares  
   - **Compression/Distortion**: JPEG compression, downscale/upscale, defocus  
   - **Occlusion**: coarse dropout, grid dropout  
3. **Loop per class**: repeatedly sample an image containing that class, apply the pipeline (with a “wrap” probability proportional to how urgently that class needs augmentation), and write out new image + remapped labels until the target is met.

### 4. Hyperparameter Search (Ray Tune + W&B)
- **Model**: `YOLO("yolo11s.pt")` (small) or `yolo11-medium.pt`  
- **Search space**:  
  ```yaml
  optimizer: ["SGD","AdamW"]
  lr0:      loguniform(5e-5,5e-2)
  lrf:      uniform(0.01,0.5)
  momentum: uniform(0.85,0.98)
  weight_decay: loguniform(1e-6,5e-4)
  # warmup_epochs/momentum/bias_lr, box/cls/dfl losses
  # hsv / rotate / translate / scale / shear / perspective
  # fliplr, mosaic, mixup, copy_paste

* **Ray Tune**:

  * 300 trials, 5-epoch grace period, 1 GPU/trial, fractional training (cosine LR, AMP, 40 epochs budget).
  * Logs streamed to Weights & Biases (optional).
* **Result**: best hyperparameters YAML persisted at `runs/detect/tune/best_hyperparameters.yaml`.

### 5. Model Training

* **Command-line** (`train.py`):

  ```bash
  yolo train \
    model=yolo11s.pt \
    data=urban_yolo_final_all.yaml \
    hyp=runs/detect/tune/best_hyperparameters.yaml \
    epochs=80 \
    imgsz=640 \
    batch=16 \
    cos_lr=True \
    amp=True \
    optimizer=AdamW \
    patience=17 \
    name=yolo11s_urban_final
  ```
* **Features**:

  * Mixed-precision (FP16) for 2× speed/memory.
  * Cosine-annealing LR with warmup (1–5 epochs).
  * Close-mosaic scheduling (disable mosaic in final 17 epochs).

### 6. Validation & Metrics

* **Fixed-threshold validation** script:

  ```bash
  yolo val \
    weights=best_medium.engine \
    data=urban_yolo_final_all.yaml \
    split=test \
    imgsz=640 \
    conf=0.2 \
    iou=0.45 \
    augment=True \
    save-json save-txt plots
  ```
* **Metrics reported**:

  * mAP@\[.50:.95], mAP\@.50, mAP\@.75
  * Precision, recall, per-class AP
  * Precision/recall curves & PR-area plots.

### 7. Real-Time Inference & Segmentation

* **TensorRT export**:

  ```bash
  yolo export \
    weights=best_medium.pt \
    format=engine \
    half \
    dynamic \
    simplify \
    imgsz=640 \
    batch=1
  ```
* **Inference scripts**:

  * **Image-only**: `process_image_combined(img)` → YOLO→SAM2→GPT/GroundingDINO refinement → masks & overlay → persist detections.
  * **Video**: `process_video(path)` → YOLO+BoT-SORT tracking → per-frame SAM segmentation → frame-by-frame JSON metadata + annotated mp4.
* **Performance**:

  * \~5–10 ms preprocess, \~15–25 ms inference, \~2–5 ms postprocess per 640×640 image on RTX 30xx (mixed-precision + TensorRT).
  * 8 MP video at 30 FPS end-to-end with tracking + segmentation.


This comprehensive workflow—from raw dataset consolidation through class-aware augmentation, large-scale hyperparameter search, rigorous training/validation, to real-time GPU-accelerated inference and segmentation—ensures Urban AI delivers both high accuracy and production-grade performance in dynamic urban environments.

## 🔧 Backend Development

The backend is developed using FastAPI, providing robust and scalable API services. It incorporates real-time processing with TensorRT for GPU acceleration and extensive monitoring and logging functionalities.

### 🤖 AI Integration Pipeline

1. **Model Initialization & Caching**  
   - On service start-up (and cached via `@lru_cache`), we load:  
     - **YOLOv11** (`ultralytics.YOLO`) with TensorRT-optimized weights (`best_medium.engine`) for ultra-fast bounding-box inference.  
     - **Segment Anything Model 2 (SAM2)** via Meta’s Hiera-B+ checkpoint (`sam2.1_hiera_base_plus.pt`), exposing both a `SAM2ImagePredictor` and an `SAM2AutomaticMaskGenerator`.  
     - **GroundingDINO** (`groundingdino_swinb_cogcoor.pth`) for phrase-conditioned box proposals on any “new” issues GPT suggests.

2. **Input Preprocessing**  
   - Read images/videos via OpenCV, auto-resize to a `MAX_DIM` of 1024px to cap memory.  
   - For video: extract frames, track objects across time using BoT-SORT (configured via a temporary YAML from `PERF_CFG`).

3. **Step 1: YOLOv11 Detection**  
   - Run `yolo.predict(...)` on BGR frames at `IMG_SZ=640`, `CONF_T=0.2`, `IOU_T=0.45`.  
   - Extract per-box: `[x1,y1,x2,y2]`, confidence, class ID/name, and assign a temporary `track_id` (frame-local for images, persistent across frames for video).

4. **Step 2: GPT-4.1 Refinement & Issue Augmentation**  
   - Collect up to the top-50 YOLO detections sorted by confidence.  
   - Build a JSON-based prompt that:  
     - Marks each detection as `keep: true|false`.  
     - Generates a one-sentence `description` and a one-sentence `solution` for every kept issue.  
     - Optionally proposes entirely **new** issues (`"new": true`) along with a minimal “dino_prompt” (1–3 COCO-style tokens) for each.  
   - Send via `openai.client.responses.create(model="gpt-4.1", …)`, parse the returned JSON array of refinement objects.

5. **Step 3: GroundingDINO for New Issues**  
   - For each GPT-proposed new issue, collect its `dino_prompt` tokens and call GroundingDINO (either local SwinB or remote 1.6-Pro API) to get coarse bounding boxes.  
   - Fallback to a full-frame box if no hit is found.

6. **Step 4: SAM2 Segmentation**  
   - Initialize `predictor.set_image(rgb)` once per image/frame.  
   - For **every** kept YOLO box **and** every new DINO box, run `predictor.predict(box=...)` to get up to 3 masks, select the “best” mask by highest overlap with the box, then:  
     - **Overlay** the mask onto the image with a soft glow + contour (via `overlay_masks`).  
     - **Encode** the mask as RLE (`mask_util.encode`) and extract polygon contours (`cv2.findContours`).

7. **Step 5: Annotation & Labeling**  
   - Draw YOLO boxes (and DINO boxes) with dynamic, non-overlapping labels (`draw_label`) to ensure legibility.  
   - Use random but stable colors per class or track ID.

8. **Step 6: Assembly & Output**  
   - Collect a final list of detections, each containing:  
     ```json
     {
       "track_id":   <int|null>,
       "class_id":   <int>,
       "class_name": <string>,
       "confidence": <float>,
       "bbox":       [x1,y1,x2,y2],
       "mask": {
         "rle":     { /* RLE-encoded mask */ },
         "polygon": [ /* [x1,y1,x2,y2,…] contours */ ]
       },
       "description": <string|null>,
       "solution":    <string|null>
     }
     ```  
   - For videos, the same pipeline runs frame-by-frame with tracking metadata attached.  
   - Persist all results (media record, frames, detections) in the database, enqueue background ingestion of RAG chunks for downstream chat queries.

---

This tightly-coupled pipeline blends ultra-fast bounding-box inference (YOLOv11) with precise mask segmentation (SAM2), semantic refinement & augmentation (GPT-4.1), and optional phrase-grounded bounding boxes (GroundingDINO) to deliver rich, actionable urban-issue annotations in both images and video.  


## 🤖 Chat & RAG Module

Urban AI’s Retrieval-Augmented Generation (RAG) chat ties together a React-based frontend, a FastAPI backend, and a pgvector-enabled PostgreSQL database to let city authorities ask natural-language questions and get context-aware, geo-filtered answers.

### 🖥️ Frontend Chat Interface
- **React chat page** with a sidebar of conversation sessions and a main chat panel.
- **Session management**:  
  - List existing sessions (`listSessions`)  
  - Start new conversations (clears history, creates new session)  
  - Delete sessions (`deleteSession`)
- **Messaging flow**:  
  - User submits a question (`sendChat`)  
  - Messages are appended locally and sent to `/api/chat`  
  - Assistant replies are streamed back, sidebar refreshed, and session ID persisted in `localStorage`.

### 🔗 Backend Chat API
- **Pydantic schemas** (`ChatRequest`, `ChatResponse`, `SessionSummary`, `ChatMessageResponse`, `SessionHistory`) validate requests and responses.
- **SQLAlchemy models**  
  - `ChatSession` & `ChatMessage` store conversation history per authority user.  
  - CRUD endpoints under `/chat` for sending messages, listing sessions, fetching history, and deleting sessions.

### 📚 RAG Service Implementation
1. **Chunk storage**  
   - **`RAGChunk`** table holds denormalized text blobs (“`<issue> detected at <location>. Description: … Suggest fix: …`”) with a `vector(1536)` embedding and optional latitude/longitude.
2. **Embedding**  
   - Uses OpenAI’s **text-embedding-3-small** via `rag_svc.embed(text)` to convert each chunk into a float vector.
3. **Retrieval**  
   - SQL query orders by `embedding.cosine_distance(query_emb)` (via pgvector), retrieving up to 4× k candidates.  
   - Applies an Haversine filter (`_within_radius`) to enforce `latitude`, `longitude`, `radius_km` constraints.  
   - Returns the top k closest, geo-filtered chunks.
4. **Ingestion pipeline**  
   - On new media upload, `ingest_media` iterates all `Detection` records, composes chunk text, embeds it, and stores `RAGChunk` entries.  
   - Triggered asynchronously via FastAPI’s `BackgroundTasks` (`enqueue_embeddings`).
5. **Context assembly & LLM call**  
   - For each incoming chat request, `_build_context` embeds the user’s query, retrieves top chunks, and concatenates them with `\n---\n` separators.  
   - The assembled context plus last 10 messages are sent as system prompts to GPT-4o.  
   - The assistant’s response is saved and returned to the frontend.

### 🗄️ Database & Docker Setup
- **PostgreSQL with pgvector**:  
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;


### Backend API Services

* **Authentication**: Robust OAuth2 and JWT token management with automatic refresh.
* **Inference APIs**: Separate endpoints for image and video processing.
* **Issue Management**: APIs to manage and view reported problems.
* **Analytics**: Metrics and performance tracking.

### Database Migrations

Urban AI uses Alembic to version and apply all database schema changes.  
– Create a new migration:  
```bash
docker compose exec web alembic revision --autogenerate -m "describe change"
```
– Apply migrations:
```bash
docker compose exec web alembic upgrade head
```

### Monitoring and Logging

Integrated OpenTelemetry with Prometheus and Grafana for performance monitoring, alongside Sentry for error tracking and logging.

## 📱 Frontend Applications

### Mobile Application (React Native + Expo)

Enables citizens to easily report urban issues via photo/video uploads with automatic or manual geolocation tagging. Users manage their submissions through an intuitive gallery interface.

### Web Admin Panel (React.js + TypeScript)

Designed for authorities to monitor and manage reported urban issues effectively through interactive maps and analytics dashboards, RAG based chat, facilitating efficient urban maintenance.

## 🐳 Docker Services Explained

Urban AI utilizes Docker extensively for streamlined deployment and management:

### Core Application Containers:

* **Web**: FastAPI application container, managing API logic and AI inference processes.
* **Database (PostgreSQL)**: Robust data storage and management.

### Monitoring and Logging:

* **Prometheus**: Aggregates metrics from services for real-time monitoring.
* **Grafana**: Visualizes collected metrics in accessible dashboards.
* **Loki & Promtail**: Collect and aggregate logs for analysis.

### GPU and Resource Monitoring:

* **DCGM-Exporter**: Monitors GPU resource usage.
* **cAdvisor & Node-Exporter**: Monitor container and node-level resource usage.

### Tracing and Analytics:

* **Tempo**: Provides distributed tracing for monitoring backend request flows.

## 🚀 Installation and Usage Guide

### Prerequisites

* Python 3.10+
* CUDA GPU (recommended)
* Node.js (18+), Expo CLI
* Docker and Docker Compose
* **SAM2.1 Base checkpoint** (for video tracking segmentation)
  Download `sam2.1_base.pt` from the Ultralytics release and place in: weights/sam2.1_base.pt

* **Ultralytics SAM2.1 Hiera B+ checkpoint** (for photo segmentation)  
Download `sam2.1_hiera_b+.pt` from Meta’s SAM2.1 repo and place in: weights/sam2.1_hiera_b+.pt

* **GroundingDINO SwinB CoGCoor checkpoint**  
Download `groundingdino_swinb_cogcoor.pth` from the GroundingDINO repo and place in: weights/groundingdino_swinb_cogcoor.pth
* (Config files for SAM2 and GroundingDINO are provided in `configs/`)


### Backend Setup

```bash
git clone <repository-url>
cd app
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
cp .env.example .env  # configure API keys, database credentials, etc.
docker compose up -d --build
```

API available at `http://localhost:8000/docs`, Grafana dashboard at `http://localhost:3000`.

### Frontend Setup

#### Web

```bash
cd web-app
cd web-app
npm install
# Configure .env.local with API URL and Google Maps key
npm run dev
```

### Mobile App (Expo + React Native)

- Built with **Expo**—run on device via **Expo Go**.  
- Features:
  - Photo/video upload with GPS or manual address picker  
  - Toggle YOLO-only or YOLO+SAM masks  
  - Gallery of past uploads, detail view with pinch-to-zoom  

**To run locally**:
1. Install Expo CLI:  
   ```bash
   npm install -g expo-cli
   ```

2. Clone and install:

   ```bash
   cd mobile
   cd urban-ai-mobile
   npm install
   ```
3. Update `config.ts` using your tunnel (set `API_BASE` to your tunnel host).
4. Start with tunnel mode and clear cache:

   ```bash
   npx expo start --clear --tunnel
   ```
5. Download **Expo Go** on your iOS device, scan the QR code, and you’re live!


## 🤝 Contributing and Development

### Testing and Development Tools

* Interactive Swagger API documentation
* Tailscale for secure remote API access
* Unit testing with pytest

### Useful Commands

```bash
# Backend
docker compose up -d --build

# Frontend (Web)
npm run dev

# Frontend (Mobile)
npx expo start --clear --tunnel
```

Urban AI aims to foster efficient, proactive management of city environments, significantly enhancing community wellbeing.

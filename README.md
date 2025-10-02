# 🏙️ Urban AI - Intelligent Urban Issue Detection System

![Urban AI Banner](https://img.shields.io/badge/AI-Powered-blue) ![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED) ![GPU](https://img.shields.io/badge/GPU-Accelerated-76B900) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192) ![Redis](https://img.shields.io/badge/Redis-DC382D) ![WebSocket](https://img.shields.io/badge/WebSocket-Real--time-orange)

Urban AI provides a comprehensive solution for efficiently identifying, segmenting, and managing urban issues such as potholes, graffiti, illegal parking, garbage, cracks, fallen trees, and others. The project uses advanced artificial intelligence techniques, including YOLOv11 for rapid object detection, GroundingDINO for phrase-grounded detection, GPT-4.1 for validation and description generation, and Segment Anything Model 2 (SAM2) for precise segmentation.

Integrated with a powerful **async backend** featuring WebSocket real-time updates, Celery task queues, Redis caching, and comprehensive monitoring, Urban AI empowers citizens to report problems seamlessly and allows authorities to manage urban maintenance effectively through web and mobile applications.

## 📑 Table of Contents

* [Project Overview](#-project-overview)
* [Key Features & Achievements](#-key-features--achievements)
* [AI Model Performance](#-ai-model-performance)
* [System Architecture](#-system-architecture)
* [AI Model Development and Training](#-ai-model-development-and-training)
* [Backend Development](#-backend-development)
  * [Async Inference Pipeline](#async-inference-pipeline)
  * [WebSocket Real-time Updates](#websocket-real-time-updates)
  * [Rate Limiting & Validation](#rate-limiting--validation)
  * [Monitoring and Logging](#monitoring-and-logging)
* [Frontend Applications](#-frontend-applications)
* [RAG System Implementation](#-rag-system-implementation)
* [Docker Services & Deployment](#-docker-services--deployment)
* [Installation and Usage Guide](#-installation-and-usage-guide)
* [Testing](#-testing)


## 🌟 Project Overview

Urban AI simplifies the reporting and management of city issues by utilizing AI-driven analyses of user-uploaded photos and videos. Reports are processed automatically, generating accurate classifications and segmentations of urban problems, significantly enhancing response times and management efficiency.

The system combines state-of-the-art computer vision models with a modern microservices architecture, featuring asynchronous processing, real-time updates, and intelligent RAG-powered chat assistance for urban management authorities.

## 🚀 Key Features & Achievements

### Core Achievements
- **150K+ training images** (88K original + 38K augmented + 14K validation + 9.5K test)
- **84.3% mAP50** accuracy on 12-class urban issue detection
- **74.3% mAP50-95** comprehensive accuracy metric
- **125 epochs** of optimized training with hyperparameter tuning
- **Real-time inference** with TensorRT acceleration
- **Async processing** with Celery task queues and WebSocket updates
- **RAG-powered chat** with hybrid search (BM25 + vector similarity)

### Technical Features
- **12-Class Detection**: Pothole, Graffiti, Garbage, Garbage Bin, Overflow, Illegal/Empty/Legal Parking, Cracks, Open/Closed Manholes, Fallen Trees
- **Multi-Modal Support**: Images (10MB max) and videos (50MB max, 60s duration)
- **Real-time Updates**: WebSocket-based live task progress
- **Distributed Processing**: GPU/CPU task separation with Celery
- **Rate Limiting**: Redis-backed per-endpoint and user-based limits
- **Role-Based Access**: Admin, authority, and user roles
- **Bilingual Support**: English and Romanian text processing
- **Geographic Analytics**: Heatmaps, hotspots, temporal patterns

## 📊 AI Model Performance

### YOLOv11 Custom Urban Model - Training Results

#### Dataset Statistics
- **Training**: 126,000 images (88K original + 38K augmented)
- **Validation**: 14,000 images
- **Test**: 9,532 images
- **Total Instances**: ~230,000 bounding boxes across all sets

#### Per-Class Test Performance

| Class | Training Instances | Test Images | Precision | Recall | mAP50 | mAP50-95 |
|-------|-------------------|-------------|-----------|--------|-------|----------|
| **Pothole** | 30,000 | 1,050 | 80.3% | 68.9% | 76.0% | 54.1% |
| **Graffiti** | 22,000 | 2,061 | 81.0% | 64.1% | 75.9% | 46.2% |
| **Garbage** | 15,000 | 594 | 94.4% | 97.0% | 97.4% | 90.9% |
| **Garbage Bin** | 24,000 | 804 | 96.3% | 96.6% | 98.2% | 92.1% |
| **Overflow** | 25,000 | 982 | 96.1% | 86.8% | 95.0% | 89.7% |
| **Parking Illegal** | 14,000 | 127 | 77.6% | 75.4% | 82.6% | 78.3% |
| **Parking Empty** | 16,000 | 96 | 92.3% | 64.8% | 70.0% | 69.0% |
| **Parking Legal** | 25,000 | 165 | 59.2% | 52.6% | 64.6% | 62.5% |
| **Crack** | 21,000 | 979 | 77.7% | 67.0% | 75.3% | 55.5% |
| **Open Manhole** | 13,000 | 569 | 91.5% | 91.4% | 95.2% | 90.2% |
| **Closed Manhole** | 15,000 | 771 | 85.6% | 74.3% | 84.2% | 77.0% |
| **Fallen Tree** | 14,000 | 684 | 92.4% | 87.1% | 94.3% | 85.9% |
| **Overall** | **234,000** | **9,532** | **85.3%** | **76.7%** | **84.3%** | **74.3%** |

#### Training Progression
- **Final Epoch (125)**: Box Loss: 0.581, Class Loss: 0.380, DFL Loss: 1.048
- **Training**: 125 epochs completed
- **Hardware**: NVIDIA GPU with mixed precision (FP16) training

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  ┌────────────────┐           ┌──────────────────────┐         │
│  │  React Web App │           │ React Native Mobile   │         │
│  └────────┬───────┘           └──────────┬───────────┘         │
└───────────┼───────────────────────────────┼─────────────────────┘
            │                               │
            ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API Gateway                              │
│  ┌──────────────┐   ┌─────────────────┐   ┌──────────────┐    │
│  │ Rate Limiter │──▶│  FastAPI Server │◀──▶│  WebSocket   │    │
│  └──────────────┘   └────────┬────────┘   └──────┬───────┘    │
└───────────────────────────────┼────────────────────┼────────────┘
                                │                    │
            ┌───────────────────┴────────┬──────────┴────┐
            ▼                            ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Processing Layer                            │
│  ┌─────────────────────────────┐  ┌─────────────────────────┐  │
│  │      GPU Worker              │  │      CPU Worker         │  │
│  │  • YOLO Detection            │  │  • Embeddings           │  │
│  │  • SAM2 Segmentation         │  │  • RAG Indexing         │  │
│  │  • GroundingDINO             │  │  • Background Tasks     │  │
│  │  • GPT-4.1 Refinement        │  │                         │  │
│  └─────────────┬───────────────┘  └──────────┬──────────────┘  │
└────────────────┼──────────────────────────────┼─────────────────┘
                 │                              │
                 ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                               │
│  ┌───────────────────────┐      ┌────────────────────────┐     │
│  │   PostgreSQL 15       │      │     Redis 8.2         │     │
│  │   + pgvector          │◀────▶│   Cache + Broker       │     │
│  └───────────────────────┘      └────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Monitoring Stack                             │
│  ┌────────┐  ┌──────────┐  ┌──────────────┐  ┌────────────┐   │
│  │ Tempo  │──▶│ Grafana  │  │ Prometheus   │  │   Sentry   │   │
│  └────────┘  └──────────┘  └──────────────┘  └────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### Technology Stack

#### Backend
- **Framework**: FastAPI with async/await support
- **Task Queue**: Celery 5.4 with Redis broker
- **Database**: PostgreSQL 15 with pgvector extension
- **Cache**: Redis 8.2 with connection pooling
- **WebSocket**: Native FastAPI WebSocket support
- **Authentication**: JWT with refresh tokens
- **Rate Limiting**: slowapi with Redis backend
- **Monitoring**: OpenTelemetry, Prometheus, Grafana, Tempo

#### AI/ML Stack
- **Object Detection**: YOLOv11 (custom trained)
- **Segmentation**: SAM2.1 Hiera B+ (*requires separate installation - see Prerequisites*)
- **Grounded Detection**: GroundingDINO SwinB
- **LLM**: GPT-4.1 for refinement and chat
- **Embeddings**: OpenAI text-embedding-3-large (3072 dims)
- **Acceleration**: TensorRT, mixed precision (FP16)

#### Frontend
- **Web**: React 18 + TypeScript + Vite + Tailwind CSS
- **Mobile**: React Native + Expo + Shopify Restyle
- **Maps**: Leaflet (web), React Native Maps (mobile)
- **Charts**: Recharts for analytics
- **Real-time**: WebSocket client for live updates

#### Infrastructure
- **Containerization**: Docker + Docker Compose
- **GPU Support**: NVIDIA CUDA 12.5 + TensorRT
- **Image**: nvcr.io/nvidia/tensorrt:25.04-py3
- **Storage**: Local volumes for weights, cache, uploads

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

* **Ray Tune**: Distributed hyperparameter search with multiple trials
  * GPU-accelerated training with mixed precision (AMP)
  * Logs streamed to Weights & Biases (optional)
* **Result**: Best hyperparameters saved for final training

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

* **TensorRT export** for optimized inference:
  ```bash
  yolo export weights=best_medium.pt format=engine half imgsz=640
  ```

* **Inference Pipeline**:
  * **Image Processing**: YOLO → SAM2 → GPT/GroundingDINO refinement → masks & overlay → database storage
  * **Video Processing**: YOLO + BoT-SORT tracking → per-frame SAM segmentation → annotated output


This comprehensive workflow—from raw dataset consolidation through class-aware augmentation, large-scale hyperparameter search, rigorous training/validation, to real-time GPU-accelerated inference and segmentation—ensures Urban AI delivers both high accuracy and production-grade performance in dynamic urban environments.

## 🔧 Backend Development

The backend is developed using FastAPI with full async/await support, providing robust and scalable API services. It features a sophisticated async inference pipeline with Celery task queues, WebSocket real-time updates, Redis-based rate limiting, and comprehensive monitoring.

### API Documentation
- **Interactive Swagger UI**: Available at `/api/docs` for testing and exploring all endpoints
- **OpenAPI Schema**: Automatically generated at `/api/openapi.json`

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

### Async Inference Pipeline

The system implements a fully asynchronous inference pipeline using Celery task queues and Redis for optimal performance:

1. **Request Flow**:
   ```
   Client → FastAPI → Validation → Celery Task → Redis Queue → Worker → Processing → Result
                ↓                                                              ↓
           WebSocket ← Redis Pub/Sub ← Progress Updates ← Worker Status
   ```

2. **Task Distribution**:
   - **GPU Queue**: Image/video inference (YOLO, SAM2, GroundingDINO)
   - **CPU Queue**: Embeddings, RAG indexing, CSV exports
   - **Default Queue**: Cleanup, maintenance tasks

3. **Worker Configuration**:
   ```python
   # GPU Worker (1 concurrent task, solo pool)
   celery -A app.core.celery_app worker -Q gpu --concurrency=1 --pool=solo

   # CPU Worker (4 concurrent tasks, prefork pool)
   celery -A app.core.celery_app worker -Q cpu,default --concurrency=4 --pool=prefork
   ```

4. **Task Monitoring**:
   - Real-time progress via Redis pub/sub
   - Prometheus metrics for task latency and throughput
   - Health checks for worker availability

### WebSocket Real-time Updates

The WebSocket implementation provides live task progress and status updates:

1. **Connection Management**:
   ```python
   # WebSocket connection with JWT authentication
   ws://localhost:8000/ws/updates?token={jwt_token}
   ```

2. **Event Types**:
   - `task_started`: Task begins processing
   - `task_progress`: Progress updates (0-100%)
   - `task_completed`: Final results ready
   - `task_failed`: Error notification

3. **Redis Bridge**:
   - Subscribes to task-specific Redis channels
   - Broadcasts updates to connected clients
   - Handles connection lifecycle and cleanup

### Rate Limiting & Validation

#### Input Validation
```python
# File size limits
MAX_IMAGE_SIZE_MB = 10
MAX_VIDEO_SIZE_MB = 50

# Resolution limits
MAX_IMAGE_DIMENSION = 4096  # 4K max
MAX_VIDEO_DIMENSION = 1920  # 1080p max
MAX_VIDEO_DURATION = 60     # seconds

# Supported extensions
IMAGE_EXTS = {".jpg", ".png", ".webp", ".heic", ...}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ...}
```

#### Rate Limiting Configuration
```python
INFERENCE_RATE_LIMITS = {
    "image": "10 per minute",   # Per user
    "video": "3 per minute",    # More resource intensive
    "status": "60 per minute",  # Status checks
    "media_list": "30 per minute"
}

USER_RATE_LIMITS = {
    "default": "100 per hour",
    "admin": "500 per hour",
    "premium": "250 per hour"
}
```

### API Endpoints

#### Authentication & User Management
```
POST   /api/auth/register          - User registration
POST   /api/auth/login             - Login with JWT
GET    /api/auth/me                - Current user info
POST   /api/auth/refresh           - Refresh access token
POST   /api/auth/logout            - Logout and revoke token
GET    /api/users                  - List users (admin only)
```

#### Async Inference Endpoints
```
POST   /api/inference/process-image     - Submit image for processing
POST   /api/inference/process-video     - Submit video for processing
GET    /api/inference/status/{task_id}  - Get task status
GET    /api/inference/result/{task_id}  - Get processing results
GET    /api/inference/media             - List processed media with filters
```

#### WebSocket Endpoint
```
WS     /ws/updates?token={token}        - Real-time task updates
```

#### Issue Management
```
GET    /api/problems                    - List all detected problems
GET    /api/problems/issues             - Filtered issue list
PATCH  /api/problems/issues/bulk_status - Bulk update status
GET    /api/issues/{id}                 - Get issue details
PATCH  /api/issues/{id}/status          - Update status (open/resolved/ignored)
PATCH  /api/issues/{id}/severity        - Update severity (low/medium/high)
PATCH  /api/issues/{id}/assign          - Assign to user (admin only)
PATCH  /api/issues/{id}/verify          - Verify issue (admin only)
```

#### Analytics Endpoints
```
GET    /api/analytics/kpis                        - Key performance indicators
GET    /api/analytics/uploads-by-day              - Daily upload statistics
GET    /api/analytics/uploads-by-user             - User activity metrics
GET    /api/analytics/detections/severity-by-day  - Severity trends over time
GET    /api/analytics/detections/top-classes      - Most common issue types
GET    /api/analytics/detections/confidence-by-class - Confidence distribution
GET    /api/analytics/geo/heatmap                 - Geographic heatmap data
GET    /api/analytics/geo/hotspots                - Issue concentration areas
GET    /api/analytics/issues/aging-buckets        - Issue resolution time analysis
GET    /api/analytics/performance/resolution-efficiency - Team performance metrics
```

#### Chat & RAG Endpoints
```
POST   /api/chat/stream                 - Stream chat response with RAG context
GET    /api/chat/sessions               - List user's chat sessions
GET    /api/chat/sessions/{id}          - Get session history
DELETE /api/chat/sessions/{id}          - Delete chat session
PATCH  /api/chat/sessions/{id}/title    - Update session title
POST   /api/rag/search                  - Hybrid RAG search
GET    /api/rag/chunk/{id}              - Get specific RAG chunk
GET    /api/rag/download-csv/{id}       - Export RAG results as CSV
```  


## 🧠 RAG System Implementation

Urban AI's Retrieval-Augmented Generation (RAG) system provides intelligent, context-aware responses about urban issues by combining hybrid search (PostgreSQL full-text search + vector similarity), cross-encoder reranking, and GPT-4.1 generation. The system enables city authorities to ask natural-language questions and get precise, geo-filtered answers.

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

### 📚 RAG Service Architecture

#### Hybrid Search Implementation

The RAG system uses a sophisticated hybrid retrieval approach combining:

1. **PostgreSQL Full-Text Search (BM25)**:
   ```sql
   -- Weighted FTS with bilingual support
   ALTER TABLE rag_chunks ADD COLUMN tsv tsvector GENERATED ALWAYS AS (
     setweight(to_tsvector('english', unaccent(class_name)), 'A') ||
     setweight(to_tsvector('romanian', unaccent(address)), 'B') ||
     setweight(to_tsvector('english', unaccent(chunk)), 'C')
   ) STORED;

   -- GIN index for fast search
   CREATE INDEX idx_rag_chunks_tsv ON rag_chunks USING GIN (tsv);
   ```

2. **Vector Similarity Search**:
   - **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)
   - **Storage**: PostgreSQL pgvector extension
   - **Distance**: Cosine similarity with HNSW indexing
   - **Query**: `embedding <=> query_vector` for KNN search

3. **Cross-Encoder Reranking**:
   ```python
   # Load cross-encoder model
   reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

   # Score and rerank candidates
   pairs = [(query, chunk.text) for chunk in candidates]
   scores = reranker.predict(pairs)
   reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
   ```

#### Query Processing Pipeline

```
User Query → Parse Filters → Hybrid Retrieval → Rerank → Context Assembly → LLM → Stream Response
     ↓             ↓                ↓               ↓            ↓              ↓
  Session ID   SQL Filters    BM25 + Vector   CrossEncoder   Top-K Chunks   GPT-4.1
```

#### RAG Query Parser Features

The `rag_query_parser` extracts the following SQL filters from natural language queries:

1. **Dynamic Filters Extracted**:
   - **severity**: low/medium/high (with English/Romanian synonym mapping)
   - **status**: open/resolved/ignored (with synonym mapping)
   - **assigned_to**: Authority username (validated against database)
   - **verified_by**: Admin username (validated against database)
   - **resolved_after/resolved_before**: Date range for resolution
   - **verified_after/verified_before**: Date range for verification
   - **sql_only**: Boolean flag for SQL-only queries

2. **Temporal Parsing**:
   - Natural language to ISO dates ("yesterday", "last week", "January 21")
   - Romanian timezone (RO_TZ) awareness
   - Relative date interpretation based on current time

3. **Synonym Normalization**:
   ```python
   # Severity mapping (English & Romanian)
   "critical"/"urgent"/"grave"/"urgente" → "high"
   "moderate"/"medii" → "medium"
   "minor"/"minore"/"usoare" → "low"

   # Status mapping
   "closed"/"fixed"/"rezolvate" → "resolved"
   "pending"/"active"/"deschise" → "open"
   ```

4. **Multi-Stage Retrieval**:
   - Stage 1: SQL filters on dynamic fields (if applicable)
   - Stage 2: BM25 text search on remaining query terms
   - Stage 3: Vector similarity search in parallel
   - Stage 4: Cross-encoder reranking of combined results
   - Stage 5: Top-K selection based on relevance scores

#### Key Features
- **Hybrid Retrieval**: Combines BM25 and vector search
- **Cross-Encoder Reranking**: Uses BERT-based model for relevance scoring
- **Context Window**: 8K tokens for GPT-4.1
- **Geographic and Temporal Filtering**: Location and date-based search

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

A user-friendly mobile app that enables citizens to report urban issues:

**Key Features**:
- **Media Upload**: Select photos/videos from device gallery or capture directly
- **Location Options**:
  - Automatic GPS location detection
  - Manual address picker with search
  - Map-based location adjustment
- **Processing Options**: Toggle SAM2 segmentation masks on/off
- **Real-time Progress**: Live processing updates via WebSocket with progress bar
- **Gallery View**:
  - Grid layout with video thumbnails
  - Processing status indicators (pending/processing/completed/failed)
  - Date stamps and address display
- **Detail View**:
  - Pinch-to-zoom for images
  - Video playback with controls
  - Issue descriptions and solutions display
- **File Validation**:
  - Max 10MB for images, 50MB for videos
  - Max 60 seconds video duration
  - Dimension checks (4096px images, 1920px videos)

**Screens**: `HomeScreen` (upload), `GalleryScreen` (history), `ProcessingScreen` (live status), `DetailScreen` (view results)

### Web Dashboard (React + TypeScript + Vite)

A comprehensive dashboard for authorities to monitor and manage urban issues:

**Key Features**:
- **Interactive Map** (`MapPage`):
  - Google Maps with dark/light theme support
  - Marker clustering for performance
  - Color-coded severity indicators
  - Info windows with issue details
  - Geohash-based filtering

- **Issue Management** (`IssuesPage`):
  - Sortable table with filtering options
  - Bulk status updates
  - Severity adjustment (low/medium/high)
  - Authority assignment (admin only)
  - Issue verification (admin only)
  - Media preview with thumbnails

- **Analytics Dashboard** (`AnalyticsPage`):
  - Key Performance Indicators (KPIs) with animations
  - Daily upload trends (area charts)
  - User activity metrics (bar charts)
  - Severity distribution over time
  - Detection source breakdown (pie charts)
  - Issue aging buckets by SLA
  - Geographic heatmaps
  - Resolution performance metrics
  - Temporal patterns analysis

- **RAG-Powered Chat** (`ChatPage`):
  - Session management (create/delete/rename)
  - Streaming responses with markdown support
  - Context chunks preview
  - Chat history persistence
  - Keyboard shortcuts (Ctrl/Cmd+K to focus)

- **Additional Pages**:
  - `ListPage`: Media gallery with filters
  - `Landing`: Public information page

**UI Components**: Custom Button, Input, Layout, ProtectedRoute, IssueModal
**Styling**: Tailwind CSS with dark mode support
**Authentication**: JWT-based with role checking (user/authority/admin)

## 🐳 Docker Services & Deployment

Urban AI utilizes a comprehensive Docker Compose setup for streamlined deployment:

### Service Architecture

```yaml
services:
  # Cache initialization
  cache-perms:
    image: busybox
    command: Fix HuggingFace cache permissions

  # Data Layer
  redis:
    image: redis:8.2-alpine
    ports: ["6379:6379"]
    config:
      - maxmemory: 2gb
      - maxmemory-policy: volatile-ttl
      - tcp-keepalive: 60
      - appendonly: yes

  db:
    image: pgvector/pgvector:15-pg15
    environment:
      - POSTGRES_DB=urbanai
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}

  # Application Layer
  web:
    build: .
    command: uvicorn app.main:app --reload
    ports: ["8000:8000"]
    environment:
      - ROLE=api
      - CELERY_BROKER_URL=redis://redis:6379/0
      - REDIS_HOST=redis

  celery-worker-gpu:
    build: .
    command: celery worker -Q gpu --concurrency=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - ROLE=worker
      - WORKER_KIND=gpu
      - CUDA_VISIBLE_DEVICES=0

  celery-worker-cpu:
    build: .
    command: celery worker -Q cpu,default --concurrency=4
    environment:
      - ROLE=worker
      - WORKER_KIND=cpu
      - CUDA_VISIBLE_DEVICES=

  # Monitoring Stack
  tempo:
    image: grafana/tempo:latest
    ports: ["4317:4317", "3200:3200"]

  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3001:3000"]

  alertmanager:
    image: prom/alertmanager:latest
    ports: ["9093:9093"]
```

### Resource Requirements

#### Minimum Configuration
- **CPU**: 4 cores
- **RAM**: 8GB
- **GPU**: NVIDIA GPU with CUDA support (6GB+ VRAM)
- **Storage**: 50GB for models and data
- **Docker**: 20.10+
- **Docker Compose**: 2.0+

#### Recommended Production
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 12GB+ VRAM
- **Storage**: 100GB+ SSD
- **OS**: Linux (Ubuntu recommended)

### Health Checks & Monitoring

```bash
# Service health checks
docker-compose ps                    # Service status
curl http://localhost:8000/healthz   # API health

# Worker health
docker-compose exec celery-gpu celery inspect ping
docker-compose exec celery-cpu celery inspect stats

# Database health
docker-compose exec db pg_isready

# Redis health
docker-compose exec redis redis-cli -a $REDIS_PASSWORD ping

# View logs
docker-compose logs -f web           # API logs
docker-compose logs -f celery-gpu    # GPU worker logs
docker-compose logs -f celery-cpu    # CPU worker logs
```

### Scaling Strategies

1. **Horizontal Scaling**:
   ```yaml
   # Scale CPU workers
   docker-compose up -d --scale celery-worker-cpu=4
   ```

2. **GPU Worker Distribution**:
   - Deploy GPU workers on separate nodes
   - Use shared Redis broker for coordination
   - Mount shared volume for model weights

3. **Database Optimization**:
   - Enable connection pooling
   - Configure read replicas for analytics
   - Use materialized views for dashboards

## 🧪 Testing

### Test Structure

The project includes comprehensive testing coverage for all major components:

```
tests/
├── unit/                    # Unit tests
│   ├── test_auth.py        # Authentication & JWT
│   ├── test_validation.py  # Input validation
│   ├── test_models.py      # Database models
│   └── test_rate_limiter.py # Rate limiting
├── integration/             # Integration tests
│   ├── test_inference_async.py  # Async inference pipeline
│   ├── test_websocket.py        # WebSocket connections
│   ├── test_celery_tasks.py     # Background tasks
│   └── test_rag_system.py       # RAG retrieval
├── e2e/                     # End-to-end tests
│   ├── test_inference_flow.py   # Complete inference workflow
│   └── test_chat_rag_flow.py    # Chat with RAG context
└── load/                    # Performance tests
    └── locustfile.py        # Load testing scenarios
```

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest -v

# Run specific test categories
pytest tests/unit -v                    # Unit tests only
pytest tests/integration -v             # Integration tests
pytest tests/e2e -v -m "not slow"      # E2E tests (skip slow)

# Run with coverage
pytest --cov=app --cov-report=html

# Run load tests
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

### Test Coverage Areas

1. **Authentication & Authorization**
   - JWT token generation/validation
   - Role-based access control
   - Token refresh and revocation
   - Rate limiting per user

2. **Inference Pipeline**
   - Image/video upload validation
   - Async task creation and monitoring
   - WebSocket real-time updates
   - Result retrieval and storage

3. **RAG System**
   - Hybrid search accuracy
   - Reranking effectiveness
   - Context assembly
   - Geographic filtering

4. **Database Operations**
   - CRUD operations
   - Concurrent updates
   - Transaction handling
   - Migration testing

5. **WebSocket Connections**
   - Authentication
   - Message broadcasting
   - Connection lifecycle
   - Error handling


## 🚀 Installation and Usage Guide

### Prerequisites

* Python 3.10+
* CUDA GPU (recommended)
* Node.js (18+), Expo CLI
* Docker and Docker Compose

### Required Dependencies (Not Included in Repository)

#### 1. **SAM2 Library Folder** (REQUIRED)
The `sam2/` folder contains the Segment Anything Model 2 library code and is essential for the segmentation functionality. Since it's not included in the repository, you must obtain it:

```bash
# Clone SAM2 repository
git clone https://github.com/facebookresearch/segment-anything-2.git sam2_temp

# Copy the sam2 folder to project root
cp -r sam2_temp/sam2 ./sam2

# Clean up
rm -rf sam2_temp
```

#### 2. **Model Checkpoints**

* **SAM2.1 Base checkpoint** (for video tracking segmentation)
  Download `sam2.1_base.pt` from the Ultralytics release and place in: `weights/sam2.1_base.pt`

* **SAM2.1 Hiera B+ checkpoint** (for photo segmentation)
  Download `sam2.1_hiera_b+.pt` from Meta's SAM2 repository and place in: `weights/sam2.1_hiera_b+.pt`

* **GroundingDINO SwinB checkpoint**
  Download `groundingdino_swinb_cogcoor.pth` from the GroundingDINO repository and place in: `weights/groundingdino_swinb_cogcoor.pth`

* **YOLOv11 Urban Model**
  The custom-trained `best_medium.pt` model is required for urban issue detection. Place in: `weights/best_medium.pt`

Note: Config files for SAM2 and GroundingDINO are already provided in the `configs/` directory.


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
5. Download **Expo Go** on your iOS device, scan the QR code, and you're live!

## 🏆 Acknowledgments

This project was built with contributions from:

- **AI Models**:
  - YOLOv11 by Ultralytics
  - SAM2 by Meta AI Research
  - GroundingDINO by IDEA Research
  - GPT-4.1 by OpenAI

- **Open Source Libraries**:
  - FastAPI for the backend framework
  - PostgreSQL and pgvector for data storage
  - Redis for caching and message brokering
  - Celery for distributed task processing
  - React and React Native for frontends

- **Special Thanks**:
  - Urban planning departments for domain expertise
  - Citizens who contributed training data
  - Open source community for invaluable tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Resources

- **API Documentation**: Interactive Swagger UI at `http://localhost:8000/docs`
- **OpenAPI Schema**: Available at `http://localhost:8000/openapi.json`

---

**Urban AI** - Empowering smarter cities through artificial intelligence 🏙️🤖

*Built with ❤️ for efficient, proactive management of city environments, significantly enhancing community wellbeing.*

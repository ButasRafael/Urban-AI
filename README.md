# 🌆 Urban AI: Comprehensive Smart City Issue Detection, Segmentation, and Reporting

Urban AI provides a complete solution for efficiently identifying, segmenting, and managing urban issues such as potholes, graffiti, illegal parking, garbage, cracks, fallen trees, and others. The project uses advanced artificial intelligence techniques, including YOLOv11 for rapid object detection, Grounding Dino—leveraging GPT-4.1’s newly identified detections to produce accurate bounding boxes where GPT alone cannot—GPT-4.1 to validate YOLO detections, identify any missed issues, and generate human-readable descriptions and remediation suggestions, and Segment Anything Model 2 (SAM2) for precise segmentation. Integrated with a powerful backend, intuitive web interface, and user-friendly mobile application, Urban AI empowers citizens to report problems seamlessly and allows authorities to manage urban maintenance effectively.

## 📑 Table of Contents

* [Project Overview](#project-overview)
* [AI Model Development and Training](#ai-model-development-and-training)
* [Backend Development](#backend-development)
* [Frontend Applications](#frontend-applications)
* [Docker Services Explained](#docker-services-explained)
* [Installation and Usage Guide](#installation-and-usage-guide)
* [Contributing and Development](#contributing-and-development)

## 🌟 Project Overview

Urban AI simplifies the reporting and management of city issues by utilizing AI-driven analyses of user-uploaded photos and videos. Reports are processed automatically, generating accurate classifications and segmentations of urban problems, significantly enhancing response times and management efficiency.

## 🧠 AI Model Development and Training

### Data Preparation and Augmentation

The unified dataset, `urban_yolo_final_all`, was created by merging multiple validated public datasets. To ensure robust model performance under various real-world conditions, extensive data augmentation was applied using Albumentations, incorporating transformations such as flips, rotations, brightness adjustments, and realistic simulations of occlusions and weather conditions (e.g., rain, fog).

### Hyperparameter Optimization

Hyperparameters were meticulously tuned using Ray Tune, targeting parameters such as optimizer choice (AdamW), learning rate schedules, and advanced augmentation methods (mosaic, mix-up, copy-paste). This optimization aimed to achieve optimal validation performance.

### Model Training and Validation

The chosen model, YOLOv11-small, offers a balance between accuracy and inference speed, trained over 80 epochs using batch sizes of 16 at a resolution of 640x640 pixels. Training utilized CUDA GPU acceleration and mixed-precision methods for efficiency. Rigorous evaluation was performed using mean Average Precision (mAP) across multiple IoU thresholds (0.5, 0.75, and 0.5-0.95).

### Real-time Detection and Segmentation

Real-time object tracking on videos was accomplished using the BoT-SORT algorithm, complemented by detailed segmentation masks generated through SAM2, providing accurate visualizations of urban issues.

## 🔧 Backend Development

The backend is developed using FastAPI, providing robust and scalable API services. It incorporates real-time processing with TensorRT for GPU acceleration and extensive monitoring and logging functionalities.

### AI Integration Pipeline

* YOLOv11 for quick issue detection.
* SAM2 for detailed segmentation.
* GroundingDINO and GPT-4.1 to refine detections, ensuring descriptive, human-readable outputs.

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

Designed for authorities to monitor and manage reported urban issues effectively through interactive maps and analytics dashboards, facilitating efficient urban maintenance.

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

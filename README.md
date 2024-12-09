# 🎥 Video Trimming and Person Tracking Application

An advanced application leveraging deep learning for intelligent video processing, face detection, and person tracking. This system automatically identifies and extracts video segments containing a target individual based on a reference image.

## ✨ Key Features

- 👤 **Advanced Face Detection** using Facenet
- 🎯 **Precise Pose Estimation** with YOLOv8n-pose
- 🔍 **Smart Re-Identification** using osnet_x1_0
- ⚡ **Adaptive Frame Skipping** for optimized processing
- 🐳 **Containerized Architecture** with Docker
- 🔌 **Full REST API Integration**

## 🚀 Performance Highlights

- ⏱️ Processes 1-hour video in ~22 minutes
- 📊 95% face matching confidence score
- 🎯 Intelligent frame retention with minimal false positives
- 💪 Handles partial occlusion through multi-model approach

## 🛠️ Technical Stack

### Backend
- Flask framework
- PyTorch for deep learning
- OpenCV for video processing
- Facenet-PyTorch for face embedding
- Ultralytics for pose estimation
- TorchReID for person re-identification

### Frontend
- React Native with Expo
- Interactive upload interface
- Real-time processing status

## 🔄 Core Workflow

### Input Processing
1. Reference image analysis (input_face.png)
   - Face embedding extraction
   - Pose keypoint processing
   - Feature extraction for ReID

2. Video Analysis (input_video.mp4)
   - Frame-by-frame face detection
   - Multi-model matching system
   - Dynamic frame retention

### Output Generation
- Trimmed video containing target segments (output_trimmed.mp4)
- Processing statistics and confidence scores

## 💻 System Requirements

- Python 3.10+
- Node.js 18+
- Docker
- NVIDIA GPU (recommended)

## 🏃‍♂️ Quick Start

1. Clone the repository
```bash
git clone https://github.com/yourusername/video-trimming-app.git
cd video-trimming-app
```

2. Launch containers
```bash
docker-compose up --build
```

3. Access the application
- Frontend: http://localhost:19000
- Upload your files:
  - Video file (input_video.mp4)
  - Reference image (input_face.png)
- Retrieve processed video: http://localhost:5000/uploads/output_trimmed.mp4

## 📡 API Reference

### Upload Endpoint
**POST** `/upload`

Request:
```json
{
  "video": "input_video.mp4",
  "face": "input_face.png"
}
```

Response:
```json
{
  "trimmedVideoUrl": "http://localhost:5000/uploads/output_trimmed.mp4"
}
```

## 🔧 Docker Configuration

### Backend
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0"]
```

### Frontend
```dockerfile
FROM node:18
WORKDIR /app
COPY . /app
RUN npm install
RUN npm install -g expo-cli
EXPOSE 19000 8081
CMD ["npx", "expo", "start"]
```

## 📊 Performance Specifications

| Hardware Component | Specification |
|-------------------|---------------|
| CPU               | Intel Core i7-12700K |
| GPU               | NVIDIA RTX 3060 (8GB VRAM) |
| RAM               | 32GB |

## 🤝 Contributing

We welcome contributions! Please feel free to submit pull requests, create issues, or suggest improvements.

## 📄 License

Released under the MIT License. See LICENSE file for detailed terms.

---

<div align="center">
Made with ❤️ for the video processing community
</div>

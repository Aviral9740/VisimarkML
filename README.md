# Visimark — Face Recognition Based Attendance System (ML & Backend)

Visimark is an **attendance marking system powered by facial recognition and liveness detection**.  
This repository contains the **machine learning and backend logic** responsible for face registration, verification, liveness checks, and attendance tracking.

The system prevents spoofing attacks (photos/videos) and ensures **secure, real-time attendance marking**.

---

## 🚀 Key Features

- **Face Recognition Attendance**
  - Identifies registered users using facial embeddings
  - Matches faces using cosine similarity

- **Liveness Detection (Anti-Spoofing)**
  - Texture analysis (blur & sharpness checks)
  - Motion detection across frames
  - Skin color consistency checks
  - Eye blink detection
  - Attendance marked only if liveness checks pass

- **Real-Time Verification API**
  - REST APIs for face registration and attendance verification
  - Supports both **base64 images** and **multipart file uploads**

- **Scalable Face Database**
  - Stores facial embeddings using MongoDB
  - Reloadable face embeddings without server restart

---

## 🧠 ML & Computer Vision Stack

- **Face Recognition Model**: FaceNet512 (DeepFace)
- **Similarity Metric**: Cosine Distance
- **Face Detection Backend**: OpenCV
- **Liveness Detection Techniques**:
  - Laplacian-based texture analysis
  - Frame-difference motion detection
  - HSV-based skin color detection
  - Haar Cascade eye-blink detection

---

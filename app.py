# ===============================
# app.py — Face Attendance API (MongoDB version)
# ===============================

import os
import base64
import io
from datetime import datetime
from collections import deque

from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
from PIL import Image
import face_recognition
from pymongo import MongoClient

# ===============================
# Configuration
# ===============================
MONGO_URI = "mongodb+srv://nikhilguptasgrr542006_db_user:%40Nik542006@cluster0.qfjep9c.mongodb.net/Sample_db?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "attendance_system"

IMAGE_FOLDER = "Attendancedir"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

LIVENESS_REQUIRED = True
MIN_CONFIDENCE = 0.6
TOLERANCE = 0.5

# ===============================
# Utility Functions
# ===============================

def decode_base64_image_to_bgr(data_uri: str):
    """Decode Base64 image (from React) into OpenCV BGR image."""
    if ',' in data_uri:
        data = data_uri.split(',')[1]
    else:
        data = data_uri
    img_bytes = base64.b64decode(data)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    np_img = np.array(pil_img)
    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return bgr_img


def filename_for(user_id, username):
    """Generate a safe filename for storing user images."""
    filename = f"{str(user_id).strip()}_{str(username).strip()}.jpg"
    return os.path.join(IMAGE_FOLDER, filename)

# ===============================
# MongoDB Setup
# ===============================
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
users_col = db["users"]
attendance_col = db["attendance"]

# ===============================
# Liveness Detection
# ===============================
class LivenessDetector:
    def __init__(self):
        self.texture_threshold = 0.15
        self.motion_threshold = 3.0
        self.frame_buffer = deque(maxlen=15)
        self.blink_counter = 0

    def check_texture_analysis(self, face_region):
        if face_region.size == 0:
            return False
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            return False
        texture_score = np.std(gray)
        return texture_score > 15

    def check_motion_detection(self, current_frame, face_location):
        try:
            x1, y1, x2, y2 = face_location
            h, w = current_frame.shape[:2]
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            if y2 <= y1 or x2 <= x1:
                return True
            current_face = current_frame[y1:y2, x1:x2]
            if current_face.size == 0:
                return True
            self.frame_buffer.append(current_face)
            if len(self.frame_buffer) < 5:
                return True
            size = (100, 100)
            prev_face = cv2.resize(self.frame_buffer[-5], size)
            curr_face = cv2.resize(self.frame_buffer[-1], size)
            diff = cv2.absdiff(
                cv2.cvtColor(prev_face, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(curr_face, cv2.COLOR_BGR2GRAY)
            )
            motion_score = np.mean(diff)
            return motion_score > self.motion_threshold
        except Exception:
            return True

    def check_color_analysis(self, face_region):
        try:
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
            return skin_ratio > 0.2
        except Exception:
            return True

    def check_eye_blink(self, face_region):
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
            if len(eyes) == 0:
                self.blink_counter += 1
            else:
                if self.blink_counter > 0:
                    self.blink_counter = 0
                    return True
            return True
        except Exception:
            return True

    def is_live(self, frame, face_location):
        try:
            x1, y1, x2, y2 = face_location
            face_region = frame[y1:y2, x1:x2]
            checks = {
                "texture": self.check_texture_analysis(face_region),
                "motion": self.check_motion_detection(frame, (x1, y1, x2, y2)),
                "color": self.check_color_analysis(face_region),
                "blink": self.check_eye_blink(face_region),
            }
            passed = sum(checks.values())
            is_live = passed >= 3
            checks["passed"] = f"{passed}/4"
            return is_live, checks
        except Exception as e:
            return False, {"error": str(e)}

# ===============================
# Face Service (Encodings + Verification)
# ===============================
class FaceService:
    def __init__(self, liveness):
        self.liveness = liveness
        self.known_encodings = []
        self.known_names = []
        self.load_known_faces()

    def load_known_faces(self):
        self.known_encodings = []
        self.known_names = []
        for user in users_col.find():
            image_path = user.get("image_path")
            name = f"{user['_id']}_{user['username']}"
            if not image_path or not os.path.exists(image_path):
                continue
            img = cv2.imread(image_path)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            enc = face_recognition.face_encodings(rgb)
            if enc:
                self.known_encodings.append(enc[0])
                self.known_names.append(name)
        return len(self.known_names)

    def verify(self, bgr_frame):
        small = cv2.resize(bgr_frame, (0, 0), None, 0.25, 0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)
        results = []

        for enc, loc in zip(encodings, locations):
            y1, x2, y2, x1 = [v * 4 for v in loc]

            if LIVENESS_REQUIRED:
                live, checks = self.liveness.is_live(bgr_frame, (x1, y1, x2, y2))
                if not live:
                    results.append({"matched": False, "reason": "spoofing", "liveness": checks})
                    continue

            if not self.known_encodings:
                results.append({"matched": False, "reason": "no_users"})
                continue

            matches = face_recognition.compare_faces(self.known_encodings, enc, tolerance=TOLERANCE)
            dists = face_recognition.face_distance(self.known_encodings, enc)
            best_idx = int(np.argmin(dists)) if len(dists) else -1

            if best_idx >= 0 and matches[best_idx] and dists[best_idx] < TOLERANCE:
                confidence = float(1.0 - dists[best_idx])
                name = self.known_names[best_idx]
                user_id, username = name.split("_", 1)
                results.append({
                    "matched": True,
                    "userId": user_id,
                    "username": username,
                    "confidence": confidence
                })
            else:
                results.append({"matched": False, "reason": "unknown"})

        return results

# ===============================
# Flask App
# ===============================
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

liveness = LivenessDetector()
face_service = FaceService(liveness)

# ===============================
# Routes
# ===============================

@app.route("/api/register", methods=["POST"])
def register_user():
    try:
        data = request.get_json(force=True)
        user_id = str(data["user_id"])
        username = data["username"]
        image_b64 = data["image"]

        image_path = filename_for(user_id, username)
        bgr = decode_base64_image_to_bgr(image_b64)
        cv2.imwrite(image_path, bgr)

        users_col.update_one(
            {"_id": user_id},
            {"$set": {"username": username, "image_path": image_path, "registeredAt": datetime.utcnow()}},
            upsert=True
        )

        count = face_service.load_known_faces()
        return jsonify({"success": True, "message": "User registered", "faces_loaded": count})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/verify", methods=["POST"])
def verify_attendance():
    try:
        data = request.get_json(force=True)
        image_b64 = data["image"]
        frame = decode_base64_image_to_bgr(image_b64)

        results = face_service.verify(frame)
        any_match = False
        for r in results:
            if r.get("matched"):
                any_match = True
                uid = r["userId"]
                uname = r["username"]
                conf = r["confidence"]
                date_str = datetime.now().strftime("%Y-%m-%d")
                existing = attendance_col.find_one({"userId": uid, "date": date_str})
                if not existing:
                    attendance_col.insert_one({
                        "userId": uid,
                        "username": uname,
                        "date": date_str,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "confidence": conf
                    })
                    r["attendance_marked"] = True
                else:
                    r["attendance_marked"] = False
                    r["message"] = "Already marked today"

        return jsonify({"success": any_match, "recognized": results})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/attendance/history", methods=["GET"])
def get_history():
    try:
        user_id = request.args.get("user_id")
        query = {}
        if user_id:
            query["userId"] = user_id
        records = list(attendance_col.find(query, {"_id": 0}))
        return jsonify({"success": True, "records": records})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    try:
        total_users = users_col.count_documents({})
        total_attendance = attendance_col.count_documents({})
        today = datetime.now().strftime("%Y-%m-%d")
        today_present = attendance_col.count_documents({"date": today})
        return jsonify({
            "ok": True,
            "stats": {
                "Total Users": total_users,
                "Total Attendance Records": total_attendance,
                "Today Present": today_present
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.route("/api/reload_faces", methods=["POST"])
def reload_faces():
    try:
        count = face_service.load_known_faces()
        liveness.frame_buffer.clear()
        return jsonify({"success": True, "message": f"Faces reloaded: {count}"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ===============================
# Run
# ===============================
if __name__ == "__main__":
    print("🚀 Starting Face Attendance Flask API with MongoDB ...")
    app.run(host="0.0.0.0", port=5000, debug=True)

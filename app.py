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
from deepface import DeepFace
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://nikhilguptasgrr542006_db_user:%40Nik542006@cluster0.qfjep9c.mongodb.net/Sample_db?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "attendance_system"

IMAGE_FOLDER = "Attendancedir"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

LIVENESS_REQUIRED = True
MIN_CONFIDENCE = 0.6
TOLERANCE = 0.4


FACE_MODEL = "Facenet512"


DETECTOR_BACKEND = "opencv"

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


def cosine_distance(embedding1, embedding2):
    """Calculate cosine distance between two embeddings."""
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-6)
    embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-6)

    similarity = np.dot(embedding1, embedding2)

    distance = 1 - similarity

    return distance


mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
users_col = db["users"]
attendance_col = db["attendance"]

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



class FaceService:
    def __init__(self, liveness):
        self.liveness = liveness
        self.known_embeddings = []
        self.known_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """Load all registered faces and generate embeddings using DeepFace."""
        print(f"Loading face database using {FACE_MODEL}...")
        self.known_embeddings = []
        self.known_names = []

        for user in users_col.find():
            image_path = user.get("image_path")
            name = f"{user['_id']}_{user['username']}"

            if not image_path or not os.path.exists(image_path):
                print(f"  ⚠ Skipping {name}: Image not found")
                continue

            try:
                # Generate embedding using DeepFace
                embedding_objs = DeepFace.represent(
                    img_path=image_path,
                    model_name=FACE_MODEL,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False  # Don't fail if face not detected
                )

                if embedding_objs and len(embedding_objs) > 0:
                    embedding = embedding_objs[0]["embedding"]
                    self.known_embeddings.append(embedding)
                    self.known_names.append(name)
                    print(f"  ✓ Loaded: {name}")
                else:
                    print(f"  ✗ No face detected in: {name}")

            except Exception as e:
                print(f"  ✗ Error loading {name}: {e}")

        print(f"✓ Loaded {len(self.known_names)} faces\n")
        return len(self.known_names)

    def verify(self, bgr_frame):
        """Verify faces in frame using DeepFace."""
        results = []

        try:
            embedding_objs = DeepFace.represent(
                img_path=bgr_frame,
                model_name=FACE_MODEL,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )

            if not embedding_objs:
                return [{"matched": False, "reason": "no_face_detected"}]

            for obj in embedding_objs:
                frame_embedding = obj["embedding"]
                facial_area = obj["facial_area"]

                x = facial_area["x"]
                y = facial_area["y"]
                w = facial_area["w"]
                h = facial_area["h"]
                x1, y1, x2, y2 = x, y, x + w, y + h

                if LIVENESS_REQUIRED:
                    live, checks = self.liveness.is_live(bgr_frame, (x1, y1, x2, y2))
                    if not live:
                        results.append({
                            "matched": False,
                            "reason": "spoofing",
                            "liveness": checks
                        })
                        continue

                if not self.known_embeddings:
                    results.append({"matched": False, "reason": "no_users"})
                    continue

                best_match_idx = -1
                best_distance = float('inf')

                for idx, known_embedding in enumerate(self.known_embeddings):
                    distance = cosine_distance(frame_embedding, known_embedding)

                    if distance < best_distance:
                        best_distance = distance
                        best_match_idx = idx

                if best_match_idx >= 0 and best_distance < TOLERANCE:
                    confidence = float(1.0 - best_distance)
                    name = self.known_names[best_match_idx]
                    user_id, username = name.split("_", 1)

                    results.append({
                        "matched": True,
                        "userId": user_id,
                        "username": username,
                        "confidence": confidence,
                        "distance": best_distance
                    })
                else:
                    results.append({
                        "matched": False,
                        "reason": "unknown",
                        "best_distance": best_distance
                    })

        except Exception as e:
            print(f"Verification error: {e}")
            results.append({
                "matched": False,
                "reason": "error",
                "message": str(e)
            })

        return results

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

liveness = LivenessDetector()
face_service = FaceService(liveness)


@app.route("/api/register", methods=["POST"])
def register_user():
    """Register a new user with face image."""
    try:
        data = request.get_json(force=True)
        user_id = str(data["user_id"])
        username = data["username"]
        image_b64 = data["image"]

        image_path = filename_for(user_id, username)
        bgr = decode_base64_image_to_bgr(image_b64)
        cv2.imwrite(image_path, bgr)

        try:
            embedding_objs = DeepFace.represent(
                img_path=image_path,
                model_name=FACE_MODEL,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True
            )

            if not embedding_objs:
                os.remove(image_path)
                return jsonify({
                    "success": False,
                    "message": "No face detected in image"
                }), 400

        except Exception as e:
            if os.path.exists(image_path):
                os.remove(image_path)
            return jsonify({
                "success": False,
                "message": f"Face detection failed: {str(e)}"
            }), 400

        users_col.update_one(
            {"_id": user_id},
            {"$set": {
                "username": username,
                "image_path": image_path,
                "registeredAt": datetime.utcnow()
            }},
            upsert=True
        )

        count = face_service.load_known_faces()

        return jsonify({
            "success": True,
            "message": "User registered successfully",
            "faces_loaded": count
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/verify", methods=["POST"])
def verify_attendance():
    """Verify face and mark attendance."""
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
                        "confidence": conf,
                        "timestamp": datetime.utcnow()
                    })
                    r["attendance_marked"] = True
                    r["message"] = "Attendance marked successfully"
                else:
                    r["attendance_marked"] = False
                    r["message"] = "Already marked today"

        return jsonify({"success": any_match, "recognized": results})

    except Exception as e:
        print(f"Verify error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/attendance/history", methods=["GET"])
def get_history():
    """Get attendance history with optional filters."""
    try:
        user_id = request.args.get("user_id")
        date = request.args.get("date")

        query = {}
        if user_id:
            query["userId"] = user_id
        if date:
            query["date"] = date

        records = list(attendance_col.find(query, {"_id": 0}).sort("timestamp", -1))

        return jsonify({
            "success": True,
            "records": records,
            "count": len(records)
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/users", methods=["GET"])
def get_users():
    """Get all registered users."""
    try:
        users = list(users_col.find({}, {"_id": 1, "username": 1, "registeredAt": 1}))

        for user in users:
            user["userId"] = str(user.pop("_id"))

        return jsonify({
            "success": True,
            "users": users,
            "count": len(users)
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint with system statistics."""
    try:
        total_users = users_col.count_documents({})
        total_attendance = attendance_col.count_documents({})
        today = datetime.now().strftime("%Y-%m-%d")
        today_present = attendance_col.count_documents({"date": today})

        return jsonify({
            "ok": True,
            "model": FACE_MODEL,
            "detector": DETECTOR_BACKEND,
            "liveness_enabled": LIVENESS_REQUIRED,
            "stats": {
                "Total Users": total_users,
                "Total Attendance Records": total_attendance,
                "Today Present": today_present,
                "Loaded Faces": len(face_service.known_embeddings)
            }
        })

    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.route("/api/reload_faces", methods=["POST"])
def reload_faces():
    """Reload all face embeddings from database."""
    try:
        count = face_service.load_known_faces()
        liveness.frame_buffer.clear()
        liveness.blink_counter = 0

        return jsonify({
            "success": True,
            "message": f"Faces reloaded successfully",
            "count": count
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/delete_user/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    """Delete a user and their data."""
    try:
        # Get user to find image path
        user = users_col.find_one({"_id": user_id})

        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404

        # Delete image file
        image_path = user.get("image_path")
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

        # Delete from database
        users_col.delete_one({"_id": user_id})

        # Delete attendance records (optional)
        # attendance_col.delete_many({"userId": user_id})

        # Reload faces
        face_service.load_known_faces()

        return jsonify({
            "success": True,
            "message": f"User {user_id} deleted successfully"
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ===============================
# Run
# ===============================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Face Attendance Flask API")
    print("=" * 60)
    print(f"📊 Model: {FACE_MODEL}")
    print(f"🔍 Detector: {DETECTOR_BACKEND}")
    print(f"🔒 Liveness: {'ENABLED' if LIVENESS_REQUIRED else 'DISABLED'}")
    print(f"📁 Image Folder: {IMAGE_FOLDER}")
    print(f"🗄️  Database: {DB_NAME}")
    print("=" * 60 + "\n")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
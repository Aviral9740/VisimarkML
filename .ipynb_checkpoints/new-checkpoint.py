import numpy as np
import dlib
import cv2
import face_recognition

print("Python version:", __import__('sys').version)
print("dlib version:", dlib.__version__)

# Test 1: Dlib with synthetic image
print("\n--- Test 1: Dlib detector ---")
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)  # ✅ convert to RGB
detector = dlib.get_frontal_face_detector()
dets = detector(test_img_rgb, 1)
print("✓ Dlib works! Detected faces:", len(dets))

# Test 2: Face recognition with your images
print("\n--- Test 2: Face recognition ---")
immodi = cv2.imread('Attendancedir/modi1.jpg')
if immodi is None:
    raise FileNotFoundError("Could not read Attendancedir/modi1.jpg — check your path!")

immodi = cv2.cvtColor(immodi, cv2.COLOR_BGR2RGB)

facloc = face_recognition.face_locations(immodi)
encodemodi = face_recognition.face_encodings(immodi)

print(f"✓ Found {len(facloc)} face(s)!")
print(f"Face location: {facloc[0] if facloc else 'None'}")
print(f"✓ Face encoding successful! Shape: {encodemodi[0].shape if encodemodi else 'None'}")

print("\n🎉 Everything is working!")
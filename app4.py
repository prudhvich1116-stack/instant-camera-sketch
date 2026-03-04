import cv2
import numpy as np
import os
from datetime import datetime

# ================== DESKTOP PATH ==================
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")

ORIGINAL_DIR = os.path.join(DESKTOP_PATH, "originals")
SKETCH_DIR = os.path.join(DESKTOP_PATH, "sketches")
OUTPUT_DIR = os.path.join(DESKTOP_PATH, "outputs")

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(SKETCH_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Images will be saved to Desktop")

# ================== FACE MODEL ==================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press C = Capture | S = Share | Q = Exit")

# ================== SKETCH FUNCTION ==================
def pencil_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    edges_inv = 255 - edges

    inverted = 255 - gray
    blur = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    final_sketch = cv2.bitwise_and(sketch, edges_inv)
    paper = np.full(gray.shape, 240, dtype=np.uint8)
    final_sketch = cv2.multiply(final_sketch, paper, scale=1 / 255)

    return final_sketch

# ================== TRACK LAST SAVED FILE ==================
last_output_path = None

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    clean_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Live Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # ================== CAPTURE ==================
    if key == ord('c'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        original_path = os.path.join(ORIGINAL_DIR, f"original_{timestamp}.png")
        sketch_path = os.path.join(SKETCH_DIR, f"sketch_{timestamp}.png")
        output_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.png")

        cv2.imwrite(original_path, clean_frame)

        sketch = pencil_sketch(clean_frame)
        cv2.imwrite(sketch_path, sketch)

        sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((clean_frame, sketch_bgr))
        cv2.imwrite(output_path, combined)

        last_output_path = output_path

        cv2.imshow("Captured Result (Original | Sketch)", combined)

        print("✅ Saved:")
        print(output_path)

    # ================== SHARE ==================
    elif key == ord('s'):
        if last_output_path and os.path.exists(last_output_path):
            print("📤 Opening image for sharing...")
            os.startfile(last_output_path)   # Windows share/open
        else:
            print("⚠️ No image to share. Capture first!")

    # ================== EXIT ==================
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

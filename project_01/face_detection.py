import cv2
import time
from pathlib import Path


# =========================
# 1. Model configuration
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_detection_yunet_2026may.onnx"

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000


# =========================
# 2. Check model
# =========================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Make sure the ONNX model is inside the models folder."
    )


# =========================
# 3. Create YuNet detector
# =========================

detector = cv2.FaceDetectorYN.create(
    str(MODEL_PATH),
    "",
    (320, 320),
    CONFIDENCE_THRESHOLD,
    NMS_THRESHOLD,
    TOP_K
)


# =========================
# 4. Open webcam
# =========================

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    raise RuntimeError("Could not open the webcam.")


# =========================
# 5. FPS variables
# =========================

previous_time = time.time()


# =========================
# 6. Main loop
# =========================

while True:

    success, frame = webcam.read()

    if not success:
        print("Could not read frame from webcam.")
        break

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Tell YuNet the actual frame size
    detector.setInputSize((width, height))

    # Detect faces
    _, faces = detector.detect(frame)


    # =========================
    # 7. Draw detections
    # =========================

    face_count = 0

    if faces is not None:

        face_count = len(faces)

        for face in faces:

            # Bounding box
            x, y, w, h = face[:4].astype(int)

            # Confidence score
            confidence = face[14]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Confidence text
            cv2.putText(
                frame,
                f"{confidence * 100:.1f}%",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    # =========================
    # 8. Calculate FPS
    # =========================

    current_time = time.time()

    fps = 1 / (current_time - previous_time)

    previous_time = current_time


    # =========================
    # 9. Display information
    # =========================

    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press ESC to exit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # =========================
    # 10. Show camera
    # =========================

    cv2.imshow("Project 01 - YuNet Face Detection", frame)


    # =========================
    # 11. Exit with ESC
    # =========================

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


# =========================
# 12. Cleanup
# =========================

webcam.release()
cv2.destroyAllWindows()
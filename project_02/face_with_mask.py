import cv2
import time
import math
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "face_detection_yunet_2026may.onnx"
MASK_PATH = BASE_DIR / "images" / "songoku.png"


# ============================================================
# 2. SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000

# Adjust these if your mask sits too high/low
MASK_SCALE = 2.5
MASK_Y_OFFSET = 0.05


# ============================================================
# 3. CHECK FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not MASK_PATH.exists():
    raise FileNotFoundError(
        f"Mask not found:\n{MASK_PATH}"
    )


# ============================================================
# 4. LOAD YUNET
# ============================================================

detector = cv2.FaceDetectorYN.create(
    str(MODEL_PATH),
    "",
    (320, 320),
    CONFIDENCE_THRESHOLD,
    NMS_THRESHOLD,
    TOP_K
)


# ============================================================
# 5. LOAD MASK
# ============================================================

mask_image = cv2.imread(
    str(MASK_PATH),
    cv2.IMREAD_UNCHANGED
)

if mask_image is None:
    raise RuntimeError("Could not load mask.")

if mask_image.shape[2] != 4:
    raise ValueError(
        "songoku.png must have a transparent background."
    )


# ============================================================
# 6. OPEN CAMERA
# ============================================================

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    raise RuntimeError("Could not open webcam.")


# ============================================================
# 7. MASK OVERLAY FUNCTION
# ============================================================

def overlay_mask(
    frame,
    mask,
    center_x,
    center_y,
    width,
    angle
):

    original_height, original_width = mask.shape[:2]

    # Keep original aspect ratio
    scale = width / original_width

    new_height = int(
        original_height * scale
    )

    resized = cv2.resize(
        mask,
        (width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Rotate mask
    # --------------------------------------------------------

    center = (
        resized.shape[1] // 2,
        resized.shape[0] // 2
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        resized,
        rotation_matrix,
        (
            resized.shape[1],
            resized.shape[0]
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    mask_height, mask_width = rotated.shape[:2]

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    x1 = int(
        center_x - mask_width / 2
    )

    y1 = int(
        center_y - mask_height / 2
    )

    x2 = x1 + mask_width
    y2 = y1 + mask_height

    frame_height, frame_width = frame.shape[:2]

    # --------------------------------------------------------
    # Clip to frame
    # --------------------------------------------------------

    frame_x1 = max(0, x1)
    frame_y1 = max(0, y1)

    frame_x2 = min(frame_width, x2)
    frame_y2 = min(frame_height, y2)

    if frame_x1 >= frame_x2 or frame_y1 >= frame_y2:
        return frame

    # Corresponding mask coordinates
    mask_x1 = frame_x1 - x1
    mask_y1 = frame_y1 - y1

    mask_x2 = mask_x1 + (
        frame_x2 - frame_x1
    )

    mask_y2 = mask_y1 + (
        frame_y2 - frame_y1
    )

    mask_crop = rotated[
        mask_y1:mask_y2,
        mask_x1:mask_x2
    ]

    # --------------------------------------------------------
    # Alpha blending
    # --------------------------------------------------------

    alpha = (
        mask_crop[:, :, 3:4] / 255.0
    )

    mask_rgb = mask_crop[:, :, :3]

    frame_region = frame[
        frame_y1:frame_y2,
        frame_x1:frame_x2
    ]

    frame[
        frame_y1:frame_y2,
        frame_x1:frame_x2
    ] = (
        alpha * mask_rgb
        +
        (1 - alpha) * frame_region
    ).astype("uint8")

    return frame


# ============================================================
# 8. FIND YOUR FACE
# ============================================================

def face_center(face):

    x, y, w, h = face[:4]

    return (
        x + w / 2,
        y + h / 2
    )


def distance(point1, point2):

    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        +
        (point1[1] - point2[1]) ** 2
    )


# ============================================================
# 9. MAIN LOOP
# ============================================================

previous_time = time.time()

my_face_center = None

while True:

    success, frame = webcam.read()

    if not success:
        break

    frame_height, frame_width = frame.shape[:2]

    detector.setInputSize(
        (frame_width, frame_height)
    )

    # --------------------------------------------------------
    # Detect ALL faces
    # --------------------------------------------------------

    _, faces = detector.detect(frame)

    face_count = 0

    if faces is not None:

        face_count = len(faces)

        # ====================================================
        # FIRST FRAME:
        # Choose the face closest to screen center
        # ====================================================

        if my_face_center is None:

            screen_center = (
                frame_width / 2,
                frame_height / 2
            )

            my_face = min(
                faces,
                key=lambda face:
                distance(
                    face_center(face),
                    screen_center
                )
            )

        # ====================================================
        # FOLLOW YOUR FACE
        # ====================================================

        else:

            my_face = min(
                faces,
                key=lambda face:
                distance(
                    face_center(face),
                    my_face_center
                )
            )

        # ----------------------------------------------------
        # Update YOUR face position
        # ----------------------------------------------------

        current_center = face_center(my_face)

        # Smooth tracking
        if my_face_center is None:

            my_face_center = current_center

        else:

            smoothing = 0.7

            my_face_center = (
                smoothing * my_face_center[0]
                +
                (1 - smoothing) * current_center[0],

                smoothing * my_face_center[1]
                +
                (1 - smoothing) * current_center[1]
            )

        # ====================================================
        # DRAW ALL FACES
        # ====================================================

        for face in faces:

            x, y, w, h = face[:4].astype(int)

            confidence = face[14]

            center = face_center(face)

            # Determine whether this is YOUR face
            is_my_face = (
                distance(
                    center,
                    my_face_center
                ) < max(w, h) * 0.5
            )

            if is_my_face:

                # ============================================
                # YOUR FACE
                # ============================================

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "YOU",
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            else:

                # ============================================
                # OTHER PEOPLE
                # ============================================

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "FACE",
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )


        # ====================================================
        # GET YOUR FACIAL LANDMARKS
        # ====================================================

        face = my_face

        # YuNet landmark positions
        right_eye = face[4:6]
        left_eye = face[6:8]

        right_eye_x, right_eye_y = right_eye
        left_eye_x, left_eye_y = left_eye

        # ----------------------------------------------------
        # Eye center
        # ----------------------------------------------------

        eye_center_x = (
            right_eye_x + left_eye_x
        ) / 2

        eye_center_y = (
            right_eye_y + left_eye_y
        ) / 2

        # ----------------------------------------------------
        # Distance between eyes
        # ----------------------------------------------------

        eye_distance = math.sqrt(
            (left_eye_x - right_eye_x) ** 2
            +
            (left_eye_y - right_eye_y) ** 2
        )

        # ----------------------------------------------------
        # Head rotation
        # ----------------------------------------------------

        angle = -math.degrees(
            math.atan2(
                left_eye_y - right_eye_y,
                left_eye_x - right_eye_x
            )
        )

        # ----------------------------------------------------
        # Mask size
        # ----------------------------------------------------

        mask_width = int(
            w * MASK_SCALE
        )

        mask_width = max(
            mask_width,
            50
        )

        # ----------------------------------------------------
        # Mask position
        # ----------------------------------------------------

        mask_center_x = eye_center_x

        mask_center_y = (
            eye_center_y
            +
            eye_distance * MASK_Y_OFFSET
        )

        # ====================================================
        # APPLY MASK ONLY TO YOUR FACE
        # ====================================================

        frame = overlay_mask(
            frame,
            mask_image,
            mask_center_x,
            mask_center_y,
            mask_width,
            angle
        )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    fps = 1 / max(
        current_time - previous_time,
        0.0001
    )

    previous_time = current_time

    # ========================================================
    # UI
    # ========================================================

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
        "ESC - Exit",
        (20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Project 01.4 - Multi Face AR",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break


# ============================================================
# CLEANUP
# ============================================================

webcam.release()
cv2.destroyAllWindows()
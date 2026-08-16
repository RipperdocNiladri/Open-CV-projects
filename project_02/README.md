# 🥷 Project 02 — Personal Face Mask with OpenCV

A real-time computer vision project built with **Python + OpenCV** that detects multiple faces from a webcam and applies a transparent image mask to the selected user's face.

The project uses the modern **YuNet face detection model** in ONNX format instead of the older Haar Cascade approach used in Project 01.

---

## 🚀 What We Built

This project combines:

* 📷 Real-time webcam capture
* 🧠 YuNet face detection
* 👥 Multi-face detection
* 🎯 Selection and tracking of the user's face
* 🥷 PNG mask overlay
* 🔄 Mask rotation based on head tilt
* 📏 Automatic mask scaling
* ↔️ Mask movement following the face
* 📊 Real-time FPS counter
* 🖼️ Transparent PNG image compositing

### How it works

```text
              Webcam
                 │
                 ▼
        ┌─────────────────┐
        │   OpenCV Frame  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ YuNet Detector  │
        └────────┬────────┘
                 │
          Detect all faces
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Selected Face      Other Faces
        │
        ▼
  Facial landmarks
        │
        ▼
 ┌──────────────────────┐
 │ Calculate:           │
 │ • Position           │
 │ • Size               │
 │ • Head rotation      │
 └──────────┬───────────┘
            │
            ▼
      🥷 Mask Overlay
            │
            ▼
       Final Webcam
          Output
```

---

# 📁 Project Structure

```text
project_02/
│
├── images/
│   └── songoku.png
│
├── models/
│   └── face_detection_yunet_2026may.onnx
│
├── face_with_mask.py
│
└── README.md
```

### `images/`

Contains the transparent PNG used as the face mask.

Current mask:

```text
songoku.png
```

The image should ideally contain a transparent background so that only the mask itself appears over the webcam feed.

### `models/`

Contains the YuNet ONNX face detection model:

```text
face_detection_yunet_2026may.onnx
```

### `face_with_mask.py`

Main Python program containing:

* Webcam handling
* YuNet face detection
* Face selection
* Facial landmark processing
* Mask positioning
* Mask scaling
* Mask rotation
* Alpha blending
* FPS calculation

---

# 🧠 Technologies Used

| Technology | Purpose                               |
| ---------- | ------------------------------------- |
| 🐍 Python  | Main programming language             |
| 👁️ OpenCV | Computer vision and webcam processing |
| 🧠 YuNet   | Face detection                        |
| 📦 ONNX    | Model format                          |
| 🖼️ PNG    | Transparent mask image                |

---

# ⚙️ Requirements

You need:

* Python 3.x
* Webcam
* OpenCV
* NumPy

The project was developed using **Python 3.13** and **OpenCV**.

---

# 🔧 Installation

## 1. Clone the repository

```bash
git clone https://github.com/RipperdocNiladri/Open-CV-projects.git
```

Enter the project directory:

```bash
cd Open-CV-projects/project_02
```

---

## 2. Install dependencies

Install OpenCV:

```bash
python -m pip install opencv-python
```

Install NumPy:

```bash
python -m pip install numpy
```

Or install both together:

```bash
python -m pip install opencv-python numpy
```

---

## 3. Verify OpenCV

Run:

```bash
python -c "import cv2; print(cv2.__version__)"
```

You should see your installed OpenCV version.

---

# ▶️ Running the Project

Make sure your terminal is inside:

```text
Open-CV-projects/project_02/
```

Then run:

```bash
python face_with_mask.py
```

Your webcam window should open.

The program will:

1. Start the webcam
2. Detect faces
3. Select the user's face
4. Track the selected face
5. Apply the `songoku.png` mask
6. Follow head movement
7. Rotate the mask with head tilt
8. Display the number of detected faces
9. Display FPS

---

# 🎛️ Mask Settings

The mask can be adjusted directly from the settings section of `face_with_mask.py`.

## 📏 `MASK_SCALE`

Controls the overall size of the mask.

```python
MASK_SCALE = 0.8
```

Increase it for a larger mask:

```python
MASK_SCALE = 0.9
```

Decrease it for a smaller mask:

```python
MASK_SCALE = 0.7
```

### Example

```text
0.6 → Smaller
0.7 → Medium-small
0.8 → Current
0.9 → Larger
1.0 → Very large
```

The mask size is calculated relative to the detected face width.

---

## ↕️ `MASK_Y_OFFSET`

Controls the vertical position of the mask.

```python
MASK_Y_OFFSET = 0.65
```

Lower value:

```python
MASK_Y_OFFSET = 0.5
```

Moves the mask upward.

Higher value:

```python
MASK_Y_OFFSET = 0.8
```

Moves the mask downward.

---

## 🔄 `ROTATION_OFFSET`

Controls the fixed rotation adjustment of the mask.

```python
ROTATION_OFFSET = 0
```

If the mask is slightly misaligned, you can adjust this value:

```python
ROTATION_OFFSET = 5
```

or:

```python
ROTATION_OFFSET = -5
```

The actual head rotation is calculated dynamically from the facial landmarks.

---

# 👁️ Facial Landmarks

YuNet provides facial landmark information along with face detection.

The project uses the two eye landmarks to calculate head tilt.

```text
        👁️────────👁️
             │
          Eye Center
```

The program calculates the angle between the eyes:

```python
dx = left_eye_x - right_eye_x
dy = left_eye_y - right_eye_y

angle = -math.degrees(
    math.atan2(dy, dx)
)
```

This allows the mask to rotate when the user tilts their head.

---

# 📐 Automatic Mask Scaling

The mask does not have a fixed size.

Its size changes according to the detected face:

```python
mask_width = int(w * MASK_SCALE)
```

Therefore:

```text
Move closer
     ↓
Face becomes larger
     ↓
Mask becomes larger


Move away
     ↓
Face becomes smaller
     ↓
Mask becomes smaller
```

This makes the overlay behave more naturally than a fixed-size image.

---

# 🥷 Multi-Face Detection

The program can detect multiple faces simultaneously.

For example:

```text
┌──────────┐       ┌──────────┐       ┌──────────┐
│  FACE    │       │   YOU    │       │  FACE    │
│ Person 1 │       │  🥷      │       │ Person 3 │
└──────────┘       └──────────┘       └──────────┘
```

Only the selected face receives the mask.

Other detected faces are displayed normally.

---

# 🎯 Face Selection

The current project selects the face closest to the center of the webcam frame.

This keeps the project relatively simple while allowing multiple faces to be detected.

The selected face is then tracked between frames using its position.

> **Note:** This is face tracking/selection, not biometric face recognition. The program does not identify a person by their identity.

---

# 🎨 Using Your Own Mask

You can replace:

```text
images/songoku.png
```

with your own transparent PNG.

For example:

```text
images/
├── songoku.png
├── sunglasses.png
└── helmet.png
```

Then change:

```python
MASK_PATH = BASE_DIR / "images" / "songoku.png"
```

to:

```python
MASK_PATH = BASE_DIR / "images" / "sunglasses.png"
```

### Recommended image format

Use:

* PNG
* Transparent background
* Good resolution
* Mask aligned approximately around the center of the image

---

# ⌨️ Controls

| Key   | Action           |
| ----- | ---------------- |
| `ESC` | Exit the program |

---

# 🧪 Current Features

* [x] Webcam face detection
* [x] Multiple face detection
* [x] YuNet ONNX model
* [x] Transparent PNG overlay
* [x] Automatic mask scaling
* [x] Mask position tracking
* [x] Head-tilt rotation
* [x] FPS counter
* [x] Face count display
* [x] Custom mask support

---

# 🔮 Future Improvements

Possible upgrades for future projects:

* 🎯 More accurate face tracking
* 🧠 Face recognition
* 🗿 3D AR masks
* 👓 Glasses overlays
* 🎭 Multiple selectable masks
* 🖱️ GUI mask selector
* 🎨 Real-time visual effects
* 📐 Better head-pose estimation
* 🧊 Perspective-based mask transformation

---

# 📚 What I Learned

This project helped me understand:

* How modern face detection models work
* How ONNX models can be used with OpenCV
* Real-time webcam processing
* Facial landmarks
* Coordinate systems
* Image scaling
* Image rotation
* Alpha blending
* Multi-face detection
* Basic object tracking
* Real-time FPS measurement

---

# 👨‍💻 Author

**Niladri Pal**

Engineering Student | Electronics & Communication Engineering

Interested in:

* 🤖 Computer Vision
* 🔌 Embedded Systems
* 🧠 AI/ML
* ⚙️ Automation
* 🚀 Robotics
* 🛰️ Space Technology

---

## ⭐ Project Status

**Status:** 🟢 Active / Learning Project

This project is part of my **Open-CV-projects** repository, where I am learning computer vision progressively through practical projects.

> **Project 02 — Personal Face Mask with OpenCV** 🥷👁️

# Hand Controls to Navigate a Computer

A computer vision script using OpenCV and MediaPipe to allow the use of hand gestures to control and navigate a PC.

## 1. Build

```
python -m venv venv && .\venv\Scripts\activate
pip install opencv-python mediapipe pynput

```

*(Note: Requires the MediaPipe `hand_landmarker.task` model file in the root directory).*

## 2. Run

```
python demo.py

```

*Output:* A 640x480 webcam window with a hand-tracking skeleton overlay. Press `q` to quit.

## 3. Inputs & Gestures (The Contract)

* **Aim:** Keep your Middle Finger extended.

* **Click / Drag:** Pinch Index + Thumb (includes a 40px drag deadzone).

* **Scroll:** Make a fist, then move your wrist like a 2D joystick.

## 4. Out of Scope

Right-clicking and multi-monitor support are intentionally excluded for this iteration to prioritize core tracking stability.

## 5. Tests

No automated test suite (`pytest`) is used; verification is performed visually via the live tracking overlay.
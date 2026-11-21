import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from cursor_indicator import CursorIndicator
from mouse import HandMouse


# cv2.flip(frame, 1) mirrors the input, which inverts MediaPipe Tasks handedness labels
MIRROR_HANDEDNESS = True


def sort_detected_hands(results, frame_w, frame_h, mirror_handedness=True):
    """Returns dict mapping 'Left' / 'Right' -> (landmarks, score, pixel_pts)."""
    raw_detections = []
    if results.hand_landmarks and results.handedness:
        for landmarks, handedness_list in zip(results.hand_landmarks, results.handedness):
            raw_label = handedness_list[0].category_name
            score = handedness_list[0].score
            if mirror_handedness:
                label = "Right" if raw_label == "Left" else "Left"
            else:
                label = raw_label
            pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
            raw_detections.append((label, landmarks, score, pts))

    # Edge-on hands often get assigned the same label; separate by horizontal screen position
    if len(raw_detections) == 2 and raw_detections[0][0] == raw_detections[1][0]:
        raw_detections.sort(key=lambda item: item[3][9][0])
        raw_detections[0] = ("Left", raw_detections[0][1], raw_detections[0][2], raw_detections[0][3])
        raw_detections[1] = ("Right", raw_detections[1][1], raw_detections[1][2], raw_detections[1][3])

    detected_hands = {}
    for label, landmarks, score, pts in raw_detections:
        if label not in detected_hands or score > detected_hands[label][1]:
            detected_hands[label] = (landmarks, score, pts)

    return detected_hands


def main():
    # 1. Initialize Overlay & Mouse Controller
    indicator = CursorIndicator(hud_corner="bottom-right")
    hand_mouse = HandMouse(default_hand="Right")

    # 2. Setup Modern MediaPipe Hand Landmarker
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # 3. Windows Camera Setup (DirectShow + request 60 FPS to reduce motion blur if supported)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int(now * 1000)

            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            detected_hands = sort_detected_hands(results, w, h, MIRROR_HANDEDNESS)

            # Run Mouse Logic & Update Floating Overlay + Corner HUD
            cursor_state = hand_mouse.update(frame, detected_hands, now)
            indicator.update(
                cursor_state,
                active_hand=hand_mouse.active_hand,
                is_active=hand_mouse.is_mouse_active,
            )

            cv2.imshow("Hand Tracker Debug (Press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hand_mouse.cleanup()
        indicator.close()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
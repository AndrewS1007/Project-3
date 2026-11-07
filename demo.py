import cv2
import mediapipe as mp
import math
import ctypes
import time
from pynput.mouse import Button, Controller
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Initialize OS Mouse Controller
mouse = Controller()

# 2. Get Windows Screen Resolution Natively
user32 = ctypes.windll.user32
SCREEN_WIDTH, SCREEN_HEIGHT = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

# 3. Setup Modern MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
landmarker = vision.HandLandmarker.create_from_options(options)

# 4. Define Hand Connections for OpenCV Drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # pinky
]

# 5. Standard Windows Camera Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# State variables
is_clicking = False
is_dragging = False # Tracks if we are actively pulling a window
drag_offset_x, drag_offset_y = 0, 0 # Tracks the teleport distance
smoothening = 2  
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Variables for the Drag Deadzone
click_lock_x, click_lock_y = 0, 0
DRAG_DEADZONE = 40  # How many pixels you have to move before a click becomes a drag

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally so it acts like a mirror
    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    frame_timestamp_ms = int(time.time() * 1000)
    
    results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    h, w, c = frame.shape

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            
            for p1, p2 in HAND_CONNECTIONS:
                cv2.line(frame, pts[p1], pts[p2], (0, 255, 180), 2)
            for px, py in pts:
                cv2.circle(frame, (px, py), 4, (0, 255, 150), cv2.FILLED)
            
            # -------------------------
            # GET RAW TARGET (MIDDLE FINGER)
            # -------------------------
            middle_tip = hand_landmarks[12]
            target_x = int(middle_tip.x * SCREEN_WIDTH)
            target_y = int(middle_tip.y * SCREEN_HEIGHT)

            # -------------------------
            # PINCH TO CLICK (INDEX + THUMB)
            # -------------------------
            ix, iy = pts[8]  
            tx, ty = pts[4]  
            distance = math.hypot(tx - ix, ty - iy)

            # Stricter thresholds so it requires a firm pinch
            PINCH_START = 22  # Lowered from 30
            PINCH_DROP = 45   # Lowered from 55

            if distance < PINCH_START:
                if not is_clicking:
                    mouse.press(Button.left)
                    is_clicking = True
                    is_dragging = False # Reset drag state on new click
                    # Lock the cursor to the exact spot the click started
                    click_lock_x, click_lock_y = target_x, target_y
                cv2.circle(frame, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
            elif distance > PINCH_DROP:
                if is_clicking:
                    mouse.release(Button.left)
                    is_clicking = False
                    is_dragging = False # Reset drag state on release

            # -------------------------
            # APPLY DRAG DEADZONE & OFFSET
            # -------------------------
            if is_clicking:
                if not is_dragging:
                    # Calculate how far the hand has moved since the click started
                    move_dist = math.hypot(target_x - click_lock_x, target_y - click_lock_y)
                    
                    # If we haven't moved far, keep the cursor completely frozen
                    if move_dist < DRAG_DEADZONE:
                        target_x, target_y = click_lock_x, click_lock_y
                    else:
                        is_dragging = True
                        # Record the exact offset gap when breaking the deadzone
                        drag_offset_x = target_x - click_lock_x
                        drag_offset_y = target_y - click_lock_y
                
                if is_dragging:
                    # Subtract the offset so the cursor doesn't snap to the new hand position
                    target_x -= drag_offset_x
                    target_y -= drag_offset_y

            # -------------------------
            # UPDATE MOUSE POSITION
            # -------------------------
            curr_x = prev_x + (target_x - prev_x) / smoothening
            curr_y = prev_y + (target_y - prev_y) / smoothening
            
            mouse.position = (curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

    cv2.imshow("Hand Tracker Debug (Press 'q' to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
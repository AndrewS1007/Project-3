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
is_double_clicking = False
is_right_clicking = False
is_dragging = False 
drag_offset_x, drag_offset_y = 0, 0 

# Scroll Variables
is_fist_scroll = False
scroll_anchor_x = 0
scroll_anchor_y = 0
SCROLL_DEADZONE = 15  

smoothening = 2  
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Variables for the Drag Deadzone
click_lock_x, click_lock_y = 0, 0
DRAG_DEADZONE = 40  

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

            # Finger Coordinates
            tx, ty = pts[4]           # Thumb tip
            ix, iy = pts[8]           # Index tip
            ix_pip, iy_pip = pts[6]   # Index middle joint (PIP)
            ix_mcp, iy_mcp = pts[5]   # Index base knuckle (MCP)
            wrist_x, wrist_y = pts[0] # Wrist

            # Distances
            dist_index = math.hypot(tx - ix, ty - iy)             # Left Click Pinch
            dist_thumb_pip = math.hypot(tx - ix_pip, ty - iy_pip) # Double Click Tap (Upper)
            dist_thumb_mcp = math.hypot(tx - ix_mcp, ty - iy_mcp) # Double Click Tap (Lower)

            # HIGHER SENSITIVITY: Standard Pinch Thresholds (Left Click)
            # Increased from 22/45 to 35/55 so the click registers lightly
            PINCH_START = 35  
            PINCH_DROP = 55   

            # Side Tap Thresholds (Double Click)
            SIDE_TAP_START = 40
            SIDE_TAP_DROP = 60

            # -------------------------
            # GESTURE BOOLEANS
            # -------------------------
            # Fist: All four fingertips below their respective middle knuckles
            is_fist = (
                pts[8][1] > pts[6][1] and 
                pts[12][1] > pts[10][1] and 
                pts[16][1] > pts[14][1] and 
                pts[20][1] > pts[18][1]
            )
            
            # Pinky Drop (Right Click): Pinky tip below midpoint, middle finger UP
            pinky_midpoint_y = (pts[19][1] + pts[18][1]) / 2.0
            is_pinky_drop = (pts[20][1] > pinky_midpoint_y) and (pts[12][1] < pts[10][1])

            # Extended Double Click Hitbox: True if thumb hits EITHER the upper or lower joint
            is_tapping_side = (dist_thumb_pip < SIDE_TAP_START) or (dist_thumb_mcp < SIDE_TAP_START)
            is_releasing_side = (dist_thumb_pip > SIDE_TAP_DROP) and (dist_thumb_mcp > SIDE_TAP_DROP)

            # -------------------------
            # FIST DETECTOR (2D Scroll Joystick)
            # -------------------------
            if is_fist and not is_clicking and not is_double_clicking and not is_right_clicking:
                if not is_fist_scroll:
                    is_fist_scroll = True
                    scroll_anchor_x = wrist_x
                    scroll_anchor_y = wrist_y 
                else:
                    delta_x = wrist_x - scroll_anchor_x
                    delta_y = wrist_y - scroll_anchor_y
                    
                    h_scroll = 0
                    v_scroll = 0
                    
                    if abs(delta_x) > SCROLL_DEADZONE:
                        speed_x = (abs(delta_x) - SCROLL_DEADZONE) / 40.0
                        dir_x = 1 if delta_x > 0 else -1 
                        h_scroll = dir_x * speed_x
                        
                    if abs(delta_y) > SCROLL_DEADZONE:
                        speed_y = (abs(delta_y) - SCROLL_DEADZONE) / 40.0
                        dir_y = -1 if delta_y > 0 else 1 
                        v_scroll = dir_y * speed_y
                        
                    if h_scroll != 0 or v_scroll != 0:
                        mouse.scroll(h_scroll, v_scroll)
                        
                    cv2.circle(frame, (scroll_anchor_x, scroll_anchor_y), 8, (0, 0, 255), cv2.FILLED)
                    cv2.line(frame, (scroll_anchor_x, scroll_anchor_y), (wrist_x, wrist_y), (0, 0, 255), 4)
            else:
                is_fist_scroll = False

            # -------------------------
            # LEFT CLICK & DRAG (Index Tip + Thumb Tip)
            # -------------------------
            if dist_index < PINCH_START and is_releasing_side and not is_fist_scroll and not is_double_clicking and not is_right_clicking:
                if not is_clicking:
                    mouse.press(Button.left)
                    is_clicking = True
                    is_dragging = False 
                    click_lock_x, click_lock_y = target_x, target_y
                cv2.circle(frame, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
            elif dist_index > PINCH_DROP:
                if is_clicking:
                    mouse.release(Button.left)
                    is_clicking = False
                    is_dragging = False 

            # -------------------------
            # INSTANT DOUBLE CLICK (Extended Side Tap)
            # -------------------------
            # Increased distance requirement (dist_index > 55) to match the new PINCH_DROP threshold
            if is_tapping_side and dist_index > 55 and not is_fist_scroll and not is_clicking and not is_right_clicking:
                if not is_double_clicking:
                    # Manually execute the double click with a 50ms hardware-realistic delay
                    mouse.press(Button.left)
                    mouse.release(Button.left)
                    time.sleep(0.05)
                    mouse.press(Button.left)
                    mouse.release(Button.left)
                    
                    is_double_clicking = True
                
                cv2.circle(frame, pts[6], 15, (255, 255, 0), cv2.FILLED) 
                cv2.circle(frame, pts[5], 15, (255, 255, 0), cv2.FILLED)
                cv2.line(frame, pts[6], pts[5], (255, 255, 0), 10)
            elif is_releasing_side:
                is_double_clicking = False
                
            # -------------------------
            # RIGHT CLICK (Pinky Drop)
            # -------------------------
            if is_pinky_drop and not is_fist_scroll and not is_clicking and not is_double_clicking:
                if not is_right_clicking:
                    mouse.click(Button.right)
                    is_right_clicking = True
                cv2.circle(frame, pts[20], 15, (255, 0, 255), cv2.FILLED)
            elif not is_pinky_drop:
                is_right_clicking = False

            # -------------------------
            # APPLY DRAG DEADZONE & OFFSET
            # -------------------------
            if is_clicking:
                if not is_dragging:
                    move_dist = math.hypot(target_x - click_lock_x, target_y - click_lock_y)
                    
                    if move_dist < DRAG_DEADZONE:
                        target_x, target_y = click_lock_x, click_lock_y
                    else:
                        is_dragging = True
                        drag_offset_x = target_x - click_lock_x
                        drag_offset_y = target_y - click_lock_y
                
                if is_dragging:
                    target_x -= drag_offset_x
                    target_y -= drag_offset_y

            # -------------------------
            # UPDATE MOUSE POSITION
            # -------------------------
            if not is_fist_scroll:
                curr_x = prev_x + (target_x - prev_x) / smoothening
                curr_y = prev_y + (target_y - prev_y) / smoothening
                
                mouse.position = (curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

    else:
        # SAFETY FALLBACK
        if is_clicking:
            mouse.release(Button.left)
            is_clicking = False
            is_dragging = False
        is_double_clicking = False
        is_right_clicking = False
        is_fist_scroll = False

    cv2.imshow("Hand Tracker Debug (Press 'q' to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
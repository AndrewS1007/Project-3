import cv2
import mediapipe as mp
import math
import ctypes
import time
from pynput.mouse import Button, Controller
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from cursor_indicator import CursorIndicator

# 1. Initialize OS Mouse Controller & Cursor Indicator
mouse = Controller()
indicator = CursorIndicator()

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

# Active Tracking Box Margins (maps inner camera region to 100% of screen)
MARGIN_X = 0.18
MARGIN_TOP = 0.25
MARGIN_BOTTOM = 0.18

# State variables
is_clicking = False
is_double_clicking = False
is_right_clicking = False
is_dragging = False
drag_offset_x, drag_offset_y = 0, 0

# Brief persistence timer for instant clicks (double-click / right-click)
flash_text = None
flash_color = None
flash_until = 0.0
FLASH_DURATION = 0.35

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

# Thumb-Across-Palm Thresholds (0.0 = index base, 1.0 = pinky base)
THUMB_CLICK_START = 0.12
THUMB_CLICK_DROP = -0.08

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

    # Draw Active Tracking Box on debug view
    box_x1 = int(MARGIN_X * w)
    box_y1 = int(MARGIN_TOP * h)
    box_x2 = int((1.0 - MARGIN_X) * w)
    box_y2 = int((1.0 - MARGIN_BOTTOM) * h)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (120, 120, 120), 1)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            for p1, p2 in HAND_CONNECTIONS:
                cv2.line(frame, pts[p1], pts[p2], (0, 255, 180), 2)
            for px, py in pts:
                cv2.circle(frame, (px, py), 4, (0, 255, 150), cv2.FILLED)

            # Highlight Landmark 9 (Middle MCP) as the active cursor anchor
            cv2.circle(frame, pts[9], 8, (255, 255, 255), 2)

            # -------------------------
            # GET RAW TARGET (MIDDLE MCP - LANDMARK 9)
            # -------------------------
            middle_mcp = hand_landmarks[9]
            norm_x = (middle_mcp.x - MARGIN_X) / (1.0 - 2.0 * MARGIN_X)
            norm_y = (middle_mcp.y - MARGIN_TOP) / (1.0 - MARGIN_TOP - MARGIN_BOTTOM)

            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))

            target_x = int(norm_x * (SCREEN_WIDTH - 1))
            target_y = int(norm_y * (SCREEN_HEIGHT - 1))

            # Key Coordinates
            tx, ty = pts[4]           # Thumb tip
            ix_mcp, iy_mcp = pts[5]   # Index base knuckle
            px_mcp, py_mcp = pts[17]  # Pinky base knuckle
            wrist_x, wrist_y = pts[0] # Wrist

            # -------------------------
            # THUMB ACROSS PALM RATIO (Scale-Invariant)
            # -------------------------
            palm_dx = px_mcp - ix_mcp
            palm_dy = py_mcp - iy_mcp
            palm_len_sq = max(palm_dx * palm_dx + palm_dy * palm_dy, 1)
            thumb_palm_ratio = ((tx - ix_mcp) * palm_dx + (ty - iy_mcp) * palm_dy) / palm_len_sq

            # -------------------------
            # GESTURE BOOLEANS
            # -------------------------
            middle_up = pts[12][1] < pts[10][1]

            # Fist: All four fingertips below their respective middle knuckles
            is_fist = (
                pts[8][1] > pts[6][1] and
                pts[12][1] > pts[10][1] and
                pts[16][1] > pts[14][1] and
                pts[20][1] > pts[18][1]
            )

            # Pinky Drop (Right Click): Pinky tip below midpoint, middle finger UP
            pinky_midpoint_y = (pts[19][1] + pts[18][1]) / 2.0
            is_pinky_drop = (pts[20][1] > pinky_midpoint_y) and middle_up

            # Index Drop (Double Click): Index tip below midpoint, middle finger UP
            index_midpoint_y = (pts[7][1] + pts[6][1]) / 2.0
            is_index_drop = (pts[8][1] > index_midpoint_y) and middle_up

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
            # LEFT CLICK & DRAG (Thumb Across Palm)
            # -------------------------
            if thumb_palm_ratio > THUMB_CLICK_START and not is_fist_scroll and not is_double_clicking and not is_right_clicking:
                if not is_clicking and middle_up:
                    mouse.press(Button.left)
                    is_clicking = True
                    is_dragging = False
                    click_lock_x, click_lock_y = target_x, target_y
                if is_clicking:
                    cv2.circle(frame, (tx, ty), 15, (0, 255, 0), cv2.FILLED)
            elif thumb_palm_ratio < THUMB_CLICK_DROP:
                if is_clicking:
                    mouse.release(Button.left)
                    is_clicking = False
                    is_dragging = False

            # -------------------------
            # INSTANT DOUBLE CLICK (Index Drop)
            # -------------------------
            if is_index_drop and not is_fist_scroll and not is_clicking and not is_right_clicking:
                if not is_double_clicking:
                    mouse.press(Button.left)
                    mouse.release(Button.left)
                    time.sleep(0.05)
                    mouse.press(Button.left)
                    mouse.release(Button.left)

                    is_double_clicking = True
                    flash_text, flash_color = "2x CLICK", "#00E5FF"
                    flash_until = time.time() + FLASH_DURATION

                cv2.circle(frame, pts[8], 15, (255, 255, 0), cv2.FILLED)
            elif not is_index_drop:
                is_double_clicking = False

            # -------------------------
            # RIGHT CLICK (Pinky Drop)
            # -------------------------
            if is_pinky_drop and not is_fist_scroll and not is_clicking and not is_double_clicking:
                if not is_right_clicking:
                    mouse.click(Button.right)
                    is_right_clicking = True
                    flash_text, flash_color = "RIGHT", "#FF40FF"
                    flash_until = time.time() + FLASH_DURATION
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

    # -------------------------
    # UPDATE FLOATING CURSOR INDICATOR
    # -------------------------
    if is_fist_scroll:
        indicator.update("SCROLL", "#FF3B30")
    elif is_clicking:
        indicator.update("DRAG" if is_dragging else "CLICK", "#00E676")
    elif is_double_clicking:
        indicator.update("2x CLICK", "#00E5FF")
    elif is_right_clicking:
        indicator.update("RIGHT", "#FF40FF")
    elif time.time() < flash_until:
        indicator.update(flash_text, flash_color)
    else:
        indicator.update(None, None)

    cv2.imshow("Hand Tracker Debug (Press 'q' to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

indicator.close()
landmarker.close()
cap.release()
cv2.destroyAllWindows()
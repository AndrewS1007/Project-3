import cv2
import numpy as np
import math
import ctypes
import time
from collections import deque
from pynput.mouse import Button, Controller


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),                  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),                  # index
    (5, 9), (9, 10), (10, 11), (11, 12),             # middle
    (9, 13), (13, 14), (14, 15), (15, 16),           # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
]


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("hCursor", ctypes.c_void_p),
        ("ptScreenPos", _POINT),
    ]


class _CursorTargetDetector:
    """Lightweight Win32 GetCursorInfo reader to detect resize bars, links, and text I-beams."""
    CURSOR_SHOWING = 0x00000001
    IDC_ARROW = 32512
    IDC_IBEAM = 32513
    IDC_WAIT = 32514
    IDC_NO = 32648
    IDC_HAND = 32649
    IDC_APPSTARTING = 32650

    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.user32.LoadCursorW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.user32.LoadCursorW.restype = ctypes.c_void_p
        self.user32.GetCursorInfo.argtypes = [ctypes.POINTER(_CURSORINFO)]
        self.user32.GetCursorInfo.restype = ctypes.c_int

        plain_ids = (
            self.IDC_ARROW,
            self.IDC_WAIT,
            self.IDC_NO,
            self.IDC_APPSTARTING,
        )
        self.plain_handles = {
            h for cid in plain_ids if (h := self.user32.LoadCursorW(None, cid))
        }
        self.ibeam_handle = self.user32.LoadCursorW(None, self.IDC_IBEAM)
        self.hand_handle = self.user32.LoadCursorW(None, self.IDC_HAND)

    def get_cursor_state(self, fallback_x, fallback_y):
        """
        Returns (target_type, screen_x, screen_y) where target_type is:
          - 'plain'  : standard arrow, busy/disabled, or hidden
          - 'ibeam'  : text input field (auto-opens keyboard on tap, no sticky friction)
          - 'hand'   : clickable link / pointer hand
          - 'resize' : standard OS resize arrows or custom app/VS Code sash handles
        """
        try:
            ci = _CURSORINFO()
            ci.cbSize = ctypes.sizeof(_CURSORINFO)
            if not self.user32.GetCursorInfo(ctypes.byref(ci)):
                return "plain", int(fallback_x), int(fallback_y)

            x, y = int(ci.ptScreenPos.x), int(ci.ptScreenPos.y)
            if not (ci.flags & self.CURSOR_SHOWING) or not ci.hCursor:
                return "plain", x, y

            if self.ibeam_handle and ci.hCursor == self.ibeam_handle:
                return "ibeam", x, y
            if ci.hCursor in self.plain_handles:
                return "plain", x, y
            if self.hand_handle and ci.hCursor == self.hand_handle:
                return "hand", x, y
            return "resize", x, y
        except Exception:
            return "plain", int(fallback_x), int(fallback_y)


def _dist(pts, a, b):
    return math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])


def get_hand_center(pts):
    """Whole-hand centroid (base knuckles + main fingertips) to capture arm or wrist swats."""
    indices = (5, 9, 13, 17, 8, 12, 16)
    cx = sum(pts[i][0] for i in indices) / len(indices)
    cy = sum(pts[i][1] for i in indices) / len(indices)
    return cx, cy


def get_palm_aspect(pts):
    """2D palm width (5-17) divided by palm length (0-9). Low values mean the hand is edge-on/sideways."""
    palm_width = math.hypot(pts[17][0] - pts[5][0], pts[17][1] - pts[5][1])
    palm_length = max(math.hypot(pts[9][0] - pts[0][0], pts[9][1] - pts[0][1]), 1.0)
    return palm_width / palm_length


def is_shaka_pose(pts):
    """
    Sensitive, rotation-invariant Shaka (thumb + pinky out; index, middle, ring curled).
    Uses wrist-relative distances so tilted hands trigger reliably without clashing with fist scroll.
    """
    palm_len = max(_dist(pts, 9, 0), 1.0)
    index_curled = _dist(pts, 8, 0) < _dist(pts, 6, 0) * 1.24
    middle_curled = _dist(pts, 12, 0) < _dist(pts, 10, 0) * 1.24
    ring_curled = _dist(pts, 16, 0) < _dist(pts, 14, 0) * 1.28
    pinky_extended = _dist(pts, 20, 0) > _dist(pts, 18, 0) * 1.07 and _dist(pts, 20, 17) > palm_len * 0.24
    thumb_extended = _dist(pts, 4, 5) > palm_len * 0.30 and _dist(pts, 4, 9) > palm_len * 0.52
    return index_curled and middle_curled and ring_curled and pinky_extended and thumb_extended


def is_fist_closed(pts):
    """Rotation-invariant closed fist: all four fingertips (including pinky) curled toward wrist."""
    return (
        _dist(pts, 8, 0) < _dist(pts, 6, 0) * 1.12
        and _dist(pts, 12, 0) < _dist(pts, 10, 0) * 1.12
        and _dist(pts, 16, 0) < _dist(pts, 14, 0) * 1.12
        and _dist(pts, 20, 0) < _dist(pts, 18, 0) * 1.10
    )


def is_fist_released(pts):
    """Hysteresis release for fist-swiping: requires at least 2 fingers clearly opened."""
    opened = sum([
        _dist(pts, 8, 0) > _dist(pts, 6, 0) * 1.20,
        _dist(pts, 12, 0) > _dist(pts, 10, 0) * 1.20,
        _dist(pts, 16, 0) > _dist(pts, 14, 0) * 1.20,
        _dist(pts, 20, 0) > _dist(pts, 18, 0) * 1.18,
    ])
    return opened >= 2


def draw_hand_skeleton(frame, pts, connections, line_color, dot_color, label_text):
    for p1, p2 in connections:
        cv2.line(frame, pts[p1], pts[p2], line_color, 2)
    for px, py in pts:
        cv2.circle(frame, (px, py), 4, dot_color, cv2.FILLED)

    label_x = max(10, pts[0][0] - 40)
    label_y = min(frame.shape[0] - 12, pts[0][1] + 24)
    cv2.putText(frame, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)


class HandMouse:
    # Active Tracking Box Margins (maps inner camera region to 100% of screen)
    MARGIN_X = 0.18
    MARGIN_TOP = 0.25
    MARGIN_BOTTOM = 0.18

    # Side-Facing Swat Handoff Settings
    SIDEWAYS_ASPECT_MAX = 0.66
    FLAT_DISARM_ASPECT = 0.74
    ARMED_GRACE_SEC = 0.45
    SWAT_WINDOW_SEC = 0.18

    # Skeleton Swat Thresholds
    SWAT_DIST_RATIO = 0.40
    SWAT_FAST_DIST_RATIO = 0.20
    SWAT_FAST_SPEED = 2.2
    SWAT_ALIGNMENT_MIN = 0.56

    # Optical Flow Swat Thresholds
    FLOW_WIDTH = 160
    FLOW_PIXEL_MIN_MAG = 1.2
    FLOW_MIN_MOVING_PIXELS = 18
    FLOW_SWAT_SPEED = 1.65
    FLOW_ALIGNMENT_MIN = 0.60

    HAND_SWITCH_COOLDOWN = 0.16
    SWITCH_FLASH_DURATION = 0.40

    # Adaptive Smoothing
    SMOOTH_SLOW = 5.0
    SMOOTH_FAST = 1.6
    KB_SWIPE_SMOOTH = 1.35         # Low-lag tracking while swiping across keyboard keys
    ADAPTIVE_SLOW_PX = 4.0
    ADAPTIVE_FAST_PX = 45.0
    JITTER_DEADZONE = 1.5

    # Precision Target Sticky Friction & Overshoot Snap-Back
    STICKY_SMOOTHENING = 8.5
    STICKY_BREAKOUT_PX = 60.0
    SNAP_MEMORY_SEC = 0.40
    SNAP_RADIUS_PX = 28.0
    SNAP_SETTLE_SEC = 0.012

    # Gesture & Drag Tuning
    ROCK_ON_HOLD_TIME = 0.25
    SHAKA_HOLD_TIME = 0.16         # Fast, responsive Shaka hold to toggle keyboard
    FIST_SCROLL_DELAY = 0.10       # Brief settle delay so transitioning into Shaka won't scroll
    FLASH_DURATION = 0.35
    SCROLL_DEADZONE = 15
    DRAG_DEADZONE = 40
    RESIZE_DRAG_DEADZONE = 14

    # Thumb-Across-Palm Thresholds (0.0 = index base, 1.0 = pinky base)
    THUMB_CLICK_START = 0.12
    THUMB_CLICK_DROP = -0.08

    def __init__(self, default_hand="Right"):
        self.mouse = Controller()
        self.cursor_detector = _CursorTargetDetector()

        user32 = ctypes.windll.user32
        self.screen_width = user32.GetSystemMetrics(0)
        self.screen_height = user32.GetSystemMetrics(1)

        # Hand-Lock & Swat Handoff State
        self.active_hand = default_hand
        self.hand_swat_history = deque()
        self.armed_until = 0.0
        self.was_armed = False
        self.armed_standby_pos = None
        self.last_active_center = None
        self.last_active_palm = 80.0
        self.pre_arm_cursor = self.mouse.position
        self.prev_gray_small = None

        self.switch_cooldown_until = 0.0
        self.switch_flash_state = None
        self.switch_flash_until = 0.0

        # Active / Pause Toggle State (Rock On gesture)
        self.is_mouse_active = True
        self.rock_on_start_time = None
        self.rock_on_triggered = False

        # Keyboard Toggle (Shaka) & Virtual Click Interception State
        self.shaka_start_time = None
        self.shaka_triggered = False
        self.keyboard_toggle_requested = False
        self.textbox_clicked = False
        self.click_started_on_ibeam = False
        self.is_virtual_clicking = False
        self.virtual_click_source = None   # "thumb", "fist", or "index"
        self.cursor_in_keyboard = False

        # Gesture State Variables
        self.is_clicking = False
        self.is_double_clicking = False
        self.is_right_clicking = False
        self.is_dragging = False
        self.drag_offset_x = 0.0
        self.drag_offset_y = 0.0

        # Brief persistence timer for instant clicks
        self.flash_state = None
        self.flash_until = 0.0

        # Scroll Variables
        self.is_fist_scroll = False
        self.fist_scroll_start_time = None
        self.scroll_anchor_x = 0
        self.scroll_anchor_y = 0

        # Cursor Smoothing, Drag Anchor & Snap-Back Memory
        self.prev_x, self.prev_y = self.mouse.position
        self.curr_x, self.curr_y = self.prev_x, self.prev_y
        self.click_lock_x = 0.0
        self.click_lock_y = 0.0
        self.click_cursor_x = 0.0
        self.click_cursor_y = 0.0
        self.active_drag_deadzone = self.DRAG_DEADZONE

        self.last_target_pos = None
        self.last_target_type = "plain"
        self.last_target_time = 0.0
        self.snap_debug_until = 0.0

    def reset_gesture_states(self):
        if self.is_clicking:
            self.mouse.release(Button.left)
        self.is_clicking = False
        self.is_virtual_clicking = False
        self.virtual_click_source = None
        self.click_started_on_ibeam = False
        self.is_dragging = False
        self.is_double_clicking = False
        self.is_right_clicking = False
        self.is_fist_scroll = False
        self.fist_scroll_start_time = None
        self.rock_on_start_time = None
        self.rock_on_triggered = False
        self.shaka_start_time = None
        self.shaka_triggered = False
        self.last_target_pos = None
        self.last_target_type = "plain"

    def cleanup(self):
        """Safely release any held mouse buttons on program exit."""
        self.reset_gesture_states()

    def _trigger_hand_switch(self, new_hand, restore_pos, now):
        self.active_hand = new_hand
        self.reset_gesture_states()
        self.hand_swat_history.clear()

        self.last_active_center, self.armed_standby_pos = self.armed_standby_pos, self.last_active_center
        self.armed_until = now + self.ARMED_GRACE_SEC
        self.was_armed = True

        self.switch_cooldown_until = now + self.HAND_SWITCH_COOLDOWN
        self.switch_flash_state = f"{self.active_hand.upper()} HAND"
        self.switch_flash_until = now + self.SWITCH_FLASH_DURATION

        if self.is_mouse_active:
            rx, ry = restore_pos
            self.mouse.position = (rx, ry)
            self.prev_x, self.prev_y = rx, ry
            self.curr_x, self.curr_y = rx, ry

    @staticmethod
    def _evaluate_swat_motion(history, target_pos):
        old_t, old_x, old_y, _ = history[0]
        last_t, last_x, last_y, last_palm = history[-1]

        to_tgt_x = target_pos[0] - old_x
        to_tgt_y = target_pos[1] - old_y
        to_tgt_dist = math.hypot(to_tgt_x, to_tgt_y)
        if to_tgt_dist < last_palm * 0.35:
            return 0.0, 0.0, 0.0

        ux, uy = to_tgt_x / to_tgt_dist, to_tgt_y / to_tgt_dist
        move_dx = last_x - old_x
        move_dy = last_y - old_y
        move_dist = math.hypot(move_dx, move_dy)
        if move_dist < 1e-6:
            return 0.0, 0.0, 0.0

        dt = max(last_t - old_t, 0.016)
        dot_toward_tgt = move_dx * ux + move_dy * uy
        projected_dist = dot_toward_tgt / last_palm
        projected_speed = projected_dist / dt
        alignment = dot_toward_tgt / move_dist
        return projected_dist, projected_speed, alignment

    def _check_optical_flow_swat(self, prev_gray, curr_gray, frame_w, frame_h, act_pos, tgt_pos, palm_size):
        flow_h, flow_w = curr_gray.shape
        sx = flow_w / float(frame_w)
        sy = flow_h / float(frame_h)

        to_tgt_x = tgt_pos[0] - act_pos[0]
        to_tgt_y = tgt_pos[1] - act_pos[1]
        to_tgt_dist = math.hypot(to_tgt_x, to_tgt_y)
        if to_tgt_dist < palm_size * 0.35:
            return False

        ux, uy = to_tgt_x / to_tgt_dist, to_tgt_y / to_tgt_dist

        r_x = max(12, int(palm_size * 0.75 * sx))
        r_y = max(12, int(palm_size * 0.75 * sy))
        cx_s = int((act_pos[0] + ux * palm_size * 0.18) * sx)
        cy_s = int((act_pos[1] + uy * palm_size * 0.18) * sy)

        x1 = max(0, cx_s - r_x)
        x2 = min(flow_w, cx_s + r_x)
        y1 = max(0, cy_s - r_y)
        y2 = min(flow_h, cy_s + r_y)
        if (x2 - x1) < 8 or (y2 - y1) < 8:
            return False

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=13,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0
        )

        roi = flow[y1:y2, x1:x2]
        fx = roi[..., 0]
        fy = roi[..., 1]
        mag = np.hypot(fx, fy)
        moving = mag > self.FLOW_PIXEL_MIN_MAG

        if np.count_nonzero(moving) < self.FLOW_MIN_MOVING_PIXELS:
            return False

        mean_fx = float(np.mean(fx[moving]))
        mean_fy = float(np.mean(fy[moving]))
        proj_flow = mean_fx * ux + mean_fy * uy
        flow_mag = math.hypot(mean_fx, mean_fy)
        alignment = proj_flow / max(flow_mag, 1e-6)

        return proj_flow >= self.FLOW_SWAT_SPEED and alignment >= self.FLOW_ALIGNMENT_MIN

    def _try_snap_back(self, cursor_type, os_x, os_y, now):
        if cursor_type in ("hand", "resize"):
            return cursor_type

        if self.last_target_pos is None or (now - self.last_target_time) > self.SNAP_MEMORY_SEC:
            return cursor_type

        sx, sy = self.last_target_pos
        if math.hypot(os_x - sx, os_y - sy) <= self.SNAP_RADIUS_PX:
            self.mouse.position = (sx, sy)
            self.curr_x, self.curr_y = float(sx), float(sy)
            self.prev_x, self.prev_y = float(sx), float(sy)
            self.snap_debug_until = now + 0.35
            time.sleep(self.SNAP_SETTLE_SEC)
            return self.last_target_type

        return cursor_type

    def _is_point_in_keyboard(self, x, y, keyboard_rect):
        return (
            keyboard_rect is not None
            and keyboard_rect[0] <= x <= keyboard_rect[2]
            and keyboard_rect[1] <= y <= keyboard_rect[3]
        )

    def update(self, frame, detected_hands, now, keyboard_rect=None):
        """
        Processes one video frame of hand detections, executes OS mouse actions,
        draws debug overlays onto `frame`, and returns the current cursor badge state.
        """
        self.keyboard_toggle_requested = False
        self.textbox_clicked = False

        h, w = frame.shape[:2]

        # Capture clean downscaled grayscale frame BEFORE drawing debug overlays
        flow_h = int(self.FLOW_WIDTH * h / w)
        gray_small = cv2.cvtColor(cv2.resize(frame, (self.FLOW_WIDTH, flow_h)), cv2.COLOR_BGR2GRAY)

        # Determine early whether the pointer is inside the keyboard overlay
        in_keyboard = self._is_point_in_keyboard(self.curr_x, self.curr_y, keyboard_rect)
        self.cursor_in_keyboard = in_keyboard
        keyboard_locked = (in_keyboard or self.is_virtual_clicking) and not self.is_clicking

        # Draw Active Tracking Box on debug view
        box_x1 = int(self.MARGIN_X * w)
        box_y1 = int(self.MARGIN_TOP * h)
        box_x2 = int((1.0 - self.MARGIN_X) * w)
        box_y2 = int((1.0 - self.MARGIN_BOTTOM) * h)
        box_color = (120, 120, 120) if self.is_mouse_active else (0, 160, 255)
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 1)

        # Draw standby hand first so active hand renders on top
        standby_hand = "Left" if self.active_hand == "Right" else "Right"
        if standby_hand in detected_hands:
            _, _, standby_pts = detected_hands[standby_hand]
            draw_hand_skeleton(
                frame, standby_pts, HAND_CONNECTIONS,
                (110, 110, 110), (140, 140, 140), f"{standby_hand.upper()} [STANDBY]"
            )

        # -------------------------
        # INSTANT SIDE-FACING SWAT HANDOFF DETECTOR (Disabled while over keyboard)
        # -------------------------
        just_switched = False

        if keyboard_locked:
            self.armed_until = 0.0
            self.hand_swat_history.clear()
            self.last_active_center = None
        elif not self.is_clicking and not self.is_fist_scroll:
            if self.active_hand in detected_hands and standby_hand in detected_hands:
                _, _, act_pts = detected_hands[self.active_hand]
                _, _, tgt_pts = detected_hands[standby_hand]
                act_aspect = get_palm_aspect(act_pts)
                tgt_aspect = get_palm_aspect(tgt_pts)

                if act_aspect < self.SIDEWAYS_ASPECT_MAX and tgt_aspect < self.SIDEWAYS_ASPECT_MAX:
                    if not self.was_armed:
                        self.pre_arm_cursor = (self.curr_x, self.curr_y)
                    self.armed_until = now + self.ARMED_GRACE_SEC
                    self.armed_standby_pos = get_hand_center(tgt_pts)
                elif act_aspect >= self.FLAT_DISARM_ASPECT:
                    self.armed_until = 0.0
            elif self.active_hand in detected_hands:
                _, _, act_pts = detected_hands[self.active_hand]
                if get_palm_aspect(act_pts) >= self.FLAT_DISARM_ASPECT:
                    self.armed_until = 0.0

        handoff_armed = (not keyboard_locked) and (now <= self.armed_until)
        can_trigger_switch = handoff_armed and (now >= self.switch_cooldown_until)

        if handoff_armed and self.armed_standby_pos is not None:
            if standby_hand in detected_hands:
                _, _, tgt_pts = detected_hands[standby_hand]
                self.armed_standby_pos = get_hand_center(tgt_pts)

            if self.active_hand in detected_hands:
                _, _, act_pts = detected_hands[self.active_hand]
                act_cx, act_cy = get_hand_center(act_pts)
                palm_size = max(math.hypot(act_pts[9][0] - act_pts[0][0], act_pts[9][1] - act_pts[0][1]), 1.0)
                self.last_active_center = (act_cx, act_cy)
                self.last_active_palm = palm_size

                if can_trigger_switch:
                    self.hand_swat_history.append((now, act_cx, act_cy, palm_size))
                    while self.hand_swat_history and (now - self.hand_swat_history[0][0]) > self.SWAT_WINDOW_SEC:
                        self.hand_swat_history.popleft()

            if self.last_active_center is not None:
                arrow_color = (0, 215, 255) if can_trigger_switch else (100, 180, 200)
                cv2.arrowedLine(
                    frame,
                    (int(self.last_active_center[0]), int(self.last_active_center[1])),
                    (int(self.armed_standby_pos[0]), int(self.armed_standby_pos[1])),
                    arrow_color, 3, tipLength=0.16
                )
                mid_x = int((self.last_active_center[0] + self.armed_standby_pos[0]) / 2) - 85
                mid_y = int((self.last_active_center[1] + self.armed_standby_pos[1]) / 2) - 14
                cv2.putText(frame, "READY - SWAT TO SWITCH", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, arrow_color, 2)

            if can_trigger_switch:
                if self.prev_gray_small is not None and self.last_active_center is not None:
                    if self._check_optical_flow_swat(
                        self.prev_gray_small, gray_small, w, h,
                        self.last_active_center, self.armed_standby_pos, self.last_active_palm
                    ):
                        self._trigger_hand_switch(standby_hand, self.pre_arm_cursor, now)
                        just_switched = True

                if not just_switched and len(self.hand_swat_history) >= 2:
                    proj_dist, proj_speed, align = self._evaluate_swat_motion(
                        self.hand_swat_history, self.armed_standby_pos
                    )
                    is_full_swat = proj_dist >= self.SWAT_DIST_RATIO
                    is_fast_burst = proj_dist >= self.SWAT_FAST_DIST_RATIO and proj_speed >= self.SWAT_FAST_SPEED

                    if align >= self.SWAT_ALIGNMENT_MIN and (is_full_swat or is_fast_burst):
                        self._trigger_hand_switch(standby_hand, self.pre_arm_cursor, now)
                        just_switched = True
        else:
            self.hand_swat_history.clear()
            self.last_active_center = None

        self.was_armed = handoff_armed

        # -------------------------
        # PROCESS ONLY THE LOCKED ACTIVE HAND
        # -------------------------
        if self.active_hand in detected_hands and not just_switched:
            hand_landmarks, _, pts = detected_hands[self.active_hand]

            draw_hand_skeleton(
                frame, pts, HAND_CONNECTIONS,
                (0, 255, 180), (0, 255, 150), f"{self.active_hand.upper()} [ACTIVE]"
            )
            cv2.circle(frame, pts[9], 8, (255, 255, 255), 2)

            if not handoff_armed:
                act_aspect = get_palm_aspect(pts)
                palm_facing_camera = act_aspect >= 0.42

                # -------------------------
                # ROCK ON DETECTOR (Disabled while hovering over keyboard)
                # -------------------------
                if keyboard_locked:
                    is_rock_on = False
                    self.rock_on_start_time = None
                    self.rock_on_triggered = False
                else:
                    is_rock_on = (
                        pts[8][1] < pts[6][1] and
                        pts[20][1] < pts[18][1] and
                        pts[12][1] > pts[10][1] and
                        pts[16][1] > pts[14][1]
                    )

                    if is_rock_on and not self.is_clicking and not self.is_fist_scroll:
                        cv2.circle(frame, pts[8], 12, (0, 160, 255), cv2.FILLED)
                        cv2.circle(frame, pts[20], 12, (0, 160, 255), cv2.FILLED)

                        if self.rock_on_start_time is None:
                            self.rock_on_start_time = now
                        elif not self.rock_on_triggered and (now - self.rock_on_start_time) >= self.ROCK_ON_HOLD_TIME:
                            self.is_mouse_active = not self.is_mouse_active
                            self.rock_on_triggered = True

                            if self.is_clicking:
                                self.mouse.release(Button.left)
                            self.is_clicking = False
                            self.is_virtual_clicking = False
                            self.virtual_click_source = None
                            self.is_dragging = False
                            self.is_double_clicking = False
                            self.is_right_clicking = False
                            self.is_fist_scroll = False
                            self.fist_scroll_start_time = None
                            self.flash_state = None
                            self.flash_until = 0.0
                            self.last_target_pos = None

                            if self.is_mouse_active:
                                self.prev_x, self.prev_y = self.mouse.position
                                self.curr_x, self.curr_y = self.prev_x, self.prev_y
                    else:
                        self.rock_on_start_time = None
                        self.rock_on_triggered = False

                # -------------------------
                # SHAKA DETECTOR (Manual Keyboard Toggle - stays active to open/close)
                # -------------------------
                is_shaka = (
                    self.is_mouse_active
                    and not is_rock_on
                    and not self.is_clicking
                    and not self.is_virtual_clicking
                    and is_shaka_pose(pts)
                )

                if is_shaka:
                    self.is_fist_scroll = False
                    self.fist_scroll_start_time = None
                    cv2.circle(frame, pts[4], 12, (0, 220, 255), cv2.FILLED)
                    cv2.circle(frame, pts[20], 12, (0, 220, 255), cv2.FILLED)
                    cv2.putText(frame, "SHAKA (KEYBOARD)", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2)

                    if self.shaka_start_time is None:
                        self.shaka_start_time = now
                    elif not self.shaka_triggered and (now - self.shaka_start_time) >= self.SHAKA_HOLD_TIME:
                        self.keyboard_toggle_requested = True
                        self.shaka_triggered = True
                else:
                    self.shaka_start_time = None
                    self.shaka_triggered = False

                if self.is_mouse_active:
                    self._process_active_gestures(
                        frame, hand_landmarks, pts, palm_facing_camera,
                        is_rock_on, is_shaka, now, keyboard_rect
                    )
        elif self.active_hand not in detected_hands:
            self.reset_gesture_states()

        # Debug window status overlay
        status_label = f"MOUSE: {'ACTIVE' if self.is_mouse_active else 'INACTIVE'} | HAND: {self.active_hand.upper()}"
        status_color = (0, 230, 118) if self.is_mouse_active else (0, 160, 255)
        cv2.putText(frame, status_label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2)

        self.prev_gray_small = gray_small
        return self._get_cursor_state(now)

    def _process_keyboard_gestures(self, frame, pts, target_x, target_y, palm_facing_camera, is_shaka):
        """
        Dedicated keyboard-hover branch: disables all OS actions (scroll, right-click,
        double-click, OS left-click, magnet snap) and handles virtual keyboard presses
        via closed fist, index-finger drop, or thumb-across-palm.
        """
        # Force-clear any non-keyboard gesture states
        self.is_fist_scroll = False
        self.fist_scroll_start_time = None
        self.is_double_clicking = False
        self.is_right_clicking = False
        self.last_target_pos = None

        cv2.putText(
            frame, "KEYBOARD MODE (OTHER ACTIONS LOCKED)", (12, 54),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 2
        )

        # Key Coordinates
        tx, ty = pts[4]
        ix_mcp, iy_mcp = pts[5]
        px_mcp, py_mcp = pts[17]

        palm_dx = px_mcp - ix_mcp
        palm_dy = py_mcp - iy_mcp
        palm_len_sq = max(palm_dx * palm_dx + palm_dy * palm_dy, 1)
        thumb_palm_ratio = ((tx - ix_mcp) * palm_dx + (ty - iy_mcp) * palm_dy) / palm_len_sq

        middle_up = pts[12][1] < pts[10][1]
        is_fist = is_fist_closed(pts) and not is_shaka

        # Index Drop (same midpoint trigger as double-click, with DIP lift hysteresis)
        index_midpoint_y = (pts[7][1] + pts[6][1]) / 2.0
        is_index_drop = (pts[8][1] > index_midpoint_y) and middle_up
        index_lifted = pts[8][1] < pts[7][1]

        if self.is_virtual_clicking:
            if self.virtual_click_source == "fist":
                if is_shaka or is_fist_released(pts):
                    self.is_virtual_clicking = False
                    self.virtual_click_source = None
                else:
                    cv2.circle(frame, pts[9], 14, (255, 220, 0), cv2.FILLED)

            elif self.virtual_click_source == "index":
                # Seamlessly promote to fist if user curled all fingers into a fist swipe
                if is_fist:
                    self.virtual_click_source = "fist"
                    cv2.circle(frame, pts[9], 14, (255, 220, 0), cv2.FILLED)
                elif is_shaka or not palm_facing_camera or not middle_up or index_lifted:
                    self.is_virtual_clicking = False
                    self.virtual_click_source = None
                else:
                    cv2.circle(frame, pts[8], 15, (255, 220, 0), cv2.FILLED)

            else:  # "thumb"
                if is_fist:
                    self.virtual_click_source = "fist"
                    cv2.circle(frame, pts[9], 14, (255, 220, 0), cv2.FILLED)
                elif is_shaka or thumb_palm_ratio < self.THUMB_CLICK_DROP or not palm_facing_camera:
                    self.is_virtual_clicking = False
                    self.virtual_click_source = None
                else:
                    cv2.circle(frame, (tx, ty), 15, (255, 220, 0), cv2.FILLED)

        elif not is_shaka:
            if is_fist:
                self.is_virtual_clicking = True
                self.virtual_click_source = "fist"
                cv2.circle(frame, pts[9], 14, (255, 220, 0), cv2.FILLED)
            elif palm_facing_camera and is_index_drop:
                self.is_virtual_clicking = True
                self.virtual_click_source = "index"
                cv2.circle(frame, pts[8], 15, (255, 220, 0), cv2.FILLED)
            elif palm_facing_camera and thumb_palm_ratio > self.THUMB_CLICK_START and middle_up:
                self.is_virtual_clicking = True
                self.virtual_click_source = "thumb"
                cv2.circle(frame, (tx, ty), 15, (255, 220, 0), cv2.FILLED)

        # Move cursor smoothly across keys (low-lag during swipes; no sticky friction)
        if not is_shaka:
            dx = target_x - self.prev_x
            dy = target_y - self.prev_y
            move_dist = math.hypot(dx, dy)

            if not self.is_virtual_clicking and move_dist < self.JITTER_DEADZONE:
                return

            if self.is_virtual_clicking:
                smooth = self.KB_SWIPE_SMOOTH
            else:
                speed_t = max(
                    0.0,
                    min(1.0, (move_dist - self.ADAPTIVE_SLOW_PX) / max(self.ADAPTIVE_FAST_PX - self.ADAPTIVE_SLOW_PX, 1.0))
                )
                smooth = self.SMOOTH_SLOW + (self.SMOOTH_FAST - self.SMOOTH_SLOW) * speed_t

            self.curr_x = self.prev_x + dx / smooth
            self.curr_y = self.prev_y + dy / smooth

            self.mouse.position = (self.curr_x, self.curr_y)
            self.prev_x, self.prev_y = self.curr_x, self.curr_y

    def _process_active_gestures(self, frame, hand_landmarks, pts, palm_facing_camera, is_rock_on, is_shaka, now, keyboard_rect):
        pose_freezing_cursor = is_rock_on or is_shaka

        # -------------------------
        # GET RAW TARGET (MIDDLE MCP - LANDMARK 9)
        # -------------------------
        middle_mcp = hand_landmarks[9]
        norm_x = (middle_mcp.x - self.MARGIN_X) / (1.0 - 2.0 * self.MARGIN_X)
        norm_y = (middle_mcp.y - self.MARGIN_TOP) / (1.0 - self.MARGIN_TOP - self.MARGIN_BOTTOM)

        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        target_x = norm_x * (self.screen_width - 1)
        target_y = norm_y * (self.screen_height - 1)

        # Route strictly to keyboard mode when hovering over the keyboard (or mid-swipe)
        in_keyboard = self._is_point_in_keyboard(self.curr_x, self.curr_y, keyboard_rect)
        self.cursor_in_keyboard = in_keyboard
        if (in_keyboard or self.is_virtual_clicking) and not self.is_clicking:
            self._process_keyboard_gestures(frame, pts, target_x, target_y, palm_facing_camera, is_shaka)
            return

        # -------------------------
        # QUERY WINDOWS CURSOR SHAPE (STICKY TARGET, SNAP MEMORY & I-BEAM)
        # -------------------------
        cursor_type, os_x, os_y = self.cursor_detector.get_cursor_state(self.curr_x, self.curr_y)
        if not self.is_clicking:
            if cursor_type in ("hand", "resize"):
                self.last_target_pos = (os_x, os_y)
                self.last_target_type = cursor_type
                self.last_target_time = now
                cv2.putText(
                    frame, f"MAGNET: {cursor_type.upper()}", (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 0), 2
                )
            elif cursor_type == "ibeam":
                cv2.putText(
                    frame, "TEXTBOX (I-BEAM)", (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 100), 2
                )
            elif now < self.snap_debug_until:
                cv2.putText(
                    frame, "SNAPPED TO TARGET", (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 180), 2
                )

        # Key Coordinates
        tx, ty = pts[4]
        ix_mcp, iy_mcp = pts[5]
        px_mcp, py_mcp = pts[17]
        wrist_x, wrist_y = pts[0]

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
        is_fist = is_fist_closed(pts) and not is_shaka

        pinky_midpoint_y = (pts[19][1] + pts[18][1]) / 2.0
        is_pinky_drop = (pts[20][1] > pinky_midpoint_y) and middle_up

        index_midpoint_y = (pts[7][1] + pts[6][1]) / 2.0
        is_index_drop = (pts[8][1] > index_midpoint_y) and middle_up

        # -------------------------
        # FIST DETECTOR (2D Scroll Joystick)
        # -------------------------
        if (
            is_fist
            and not self.is_clicking
            and not self.is_double_clicking
            and not self.is_right_clicking
        ):
            if not self.is_fist_scroll:
                if self.fist_scroll_start_time is None:
                    self.fist_scroll_start_time = now
                elif (now - self.fist_scroll_start_time) >= self.FIST_SCROLL_DELAY:
                    self.is_fist_scroll = True
                    self.scroll_anchor_x = wrist_x
                    self.scroll_anchor_y = wrist_y
            else:
                delta_x = wrist_x - self.scroll_anchor_x
                delta_y = wrist_y - self.scroll_anchor_y

                h_scroll = 0
                v_scroll = 0

                if abs(delta_x) > self.SCROLL_DEADZONE:
                    speed_x = (abs(delta_x) - self.SCROLL_DEADZONE) / 40.0
                    dir_x = 1 if delta_x > 0 else -1
                    h_scroll = dir_x * speed_x

                if abs(delta_y) > self.SCROLL_DEADZONE:
                    speed_y = (abs(delta_y) - self.SCROLL_DEADZONE) / 40.0
                    dir_y = -1 if delta_y > 0 else 1
                    v_scroll = dir_y * speed_y

                if h_scroll != 0 or v_scroll != 0:
                    self.mouse.scroll(h_scroll, v_scroll)

                cv2.circle(frame, (self.scroll_anchor_x, self.scroll_anchor_y), 8, (0, 0, 255), cv2.FILLED)
                cv2.line(frame, (self.scroll_anchor_x, self.scroll_anchor_y), (wrist_x, wrist_y), (0, 0, 255), 4)
        else:
            self.is_fist_scroll = False
            self.fist_scroll_start_time = None

        # -------------------------
        # OS LEFT CLICK & DRAG (Thumb Across Palm)
        # -------------------------
        if (
            palm_facing_camera
            and thumb_palm_ratio > self.THUMB_CLICK_START
            and not self.is_fist_scroll
            and not self.is_double_clicking
            and not self.is_right_clicking
        ):
            if not self.is_clicking and middle_up:
                effective_type = self._try_snap_back(cursor_type, os_x, os_y, now)
                self.mouse.press(Button.left)
                self.is_clicking = True
                self.is_dragging = False
                self.click_started_on_ibeam = (effective_type == "ibeam")
                self.click_lock_x, self.click_lock_y = target_x, target_y
                self.click_cursor_x, self.click_cursor_y = self.curr_x, self.curr_y
                self.active_drag_deadzone = (
                    self.RESIZE_DRAG_DEADZONE if effective_type == "resize" else self.DRAG_DEADZONE
                )
            if self.is_clicking:
                cv2.circle(frame, (tx, ty), 15, (0, 255, 0), cv2.FILLED)
        elif thumb_palm_ratio < self.THUMB_CLICK_DROP or not palm_facing_camera:
            if self.is_clicking:
                was_dragging = self.is_dragging
                self.mouse.release(Button.left)
                self.is_clicking = False
                self.is_dragging = False

                if self.click_started_on_ibeam and not was_dragging and keyboard_rect is None:
                    self.textbox_clicked = True
                self.click_started_on_ibeam = False

        # -------------------------
        # INSTANT DOUBLE CLICK (Index Drop)
        # -------------------------
        if (
            palm_facing_camera
            and is_index_drop
            and not self.is_fist_scroll
            and not self.is_clicking
            and not self.is_right_clicking
        ):
            if not self.is_double_clicking:
                self.mouse.press(Button.left)
                self.mouse.release(Button.left)
                time.sleep(0.05)
                self.mouse.press(Button.left)
                self.mouse.release(Button.left)

                self.is_double_clicking = True
                self.flash_state = "2x CLICK"
                self.flash_until = now + self.FLASH_DURATION

            cv2.circle(frame, pts[8], 15, (255, 255, 0), cv2.FILLED)
        elif not is_index_drop:
            self.is_double_clicking = False

        # -------------------------
        # RIGHT CLICK (Pinky Drop)
        # -------------------------
        if (
            palm_facing_camera
            and is_pinky_drop
            and not self.is_fist_scroll
            and not self.is_clicking
            and not self.is_double_clicking
        ):
            if not self.is_right_clicking:
                self.mouse.click(Button.right)
                self.is_right_clicking = True
                self.flash_state = "RIGHT"
                self.flash_until = now + self.FLASH_DURATION
            cv2.circle(frame, pts[20], 15, (255, 0, 255), cv2.FILLED)
        elif not is_pinky_drop:
            self.is_right_clicking = False

        # -------------------------
        # APPLY DRAG DEADZONE & OFFSET (OS CLICKS)
        # -------------------------
        if self.is_clicking:
            if not self.is_dragging:
                move_dist = math.hypot(target_x - self.click_lock_x, target_y - self.click_lock_y)

                if move_dist < self.active_drag_deadzone:
                    target_x, target_y = self.click_cursor_x, self.click_cursor_y
                else:
                    self.is_dragging = True
                    self.drag_offset_x = target_x - self.click_cursor_x
                    self.drag_offset_y = target_y - self.click_cursor_y

            if self.is_dragging:
                target_x -= self.drag_offset_x
                target_y -= self.drag_offset_y

        # -------------------------
        # UPDATE MOUSE POSITION (ADAPTIVE SMOOTHING + STICKY FRICTION)
        # -------------------------
        if not self.is_fist_scroll and not pose_freezing_cursor:
            if self.is_clicking and not self.is_dragging:
                self.curr_x, self.curr_y = self.click_cursor_x, self.click_cursor_y
                self.prev_x, self.prev_y = self.curr_x, self.curr_y
                return

            dx = target_x - self.prev_x
            dy = target_y - self.prev_y
            move_dist = math.hypot(dx, dy)

            if not self.is_dragging and move_dist < self.JITTER_DEADZONE:
                return

            speed_t = max(
                0.0,
                min(1.0, (move_dist - self.ADAPTIVE_SLOW_PX) / max(self.ADAPTIVE_FAST_PX - self.ADAPTIVE_SLOW_PX, 1.0))
            )
            smooth = self.SMOOTH_SLOW + (self.SMOOTH_FAST - self.SMOOTH_SLOW) * speed_t

            if (
                not self.is_clicking
                and cursor_type in ("hand", "resize")
                and move_dist < self.STICKY_BREAKOUT_PX
            ):
                smooth = max(smooth, self.STICKY_SMOOTHENING)

            self.curr_x = self.prev_x + dx / smooth
            self.curr_y = self.prev_y + dy / smooth

            self.mouse.position = (self.curr_x, self.curr_y)
            self.prev_x, self.prev_y = self.curr_x, self.curr_y

    def _get_cursor_state(self, now):
        if now < self.switch_flash_until:
            return self.switch_flash_state
        if not self.is_mouse_active:
            return "INACTIVE"
        if self.cursor_in_keyboard or self.is_virtual_clicking:
            return None
        if self.is_fist_scroll:
            return "SCROLL"
        if self.is_clicking:
            return "DRAG" if self.is_dragging else "CLICK"
        if self.is_double_clicking:
            return "2x CLICK"
        if self.is_right_clicking:
            return "RIGHT"
        if now < self.flash_until:
            return self.flash_state
        return None
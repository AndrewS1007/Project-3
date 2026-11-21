import ctypes
import math
import os
import time
import tkinter as tk
import numpy as np
from pynput.keyboard import Controller as KeyController, Key


# Compact frequency-ordered starter vocabulary for immediate swipe testing.
# Dropping a `wordlist.txt` (one word per line, ranked by frequency) in the
# project folder will automatically replace this with a full dictionary.
STARTER_WORDS = (
    "the be to of and in that have it for not on with he as you do at this but "
    "his by from they we say her she or an will my one all would there their what "
    "so up out if about who get which go me when make can like time no just him "
    "know take people into year your good some could them see other than then now "
    "look only come its over think also back after use two how our work first well "
    "way even new want because any these give day most us are was were been has "
    "had did does doing done hello world test swipe type typing keyboard mouse hand "
    "track tracking open close save run code python file folder project window "
    "screen click drag scroll left right top bottom up down start stop pause play "
    "help please thanks thank sorry yes yeah okay sure great awesome nice cool "
    "good morning night today tomorrow yesterday week month year home school work "
    "class study note notes email message chat send reply check read write edit "
    "delete copy paste undo redo search find next prev back enter space shift "
    "here where why how much many more less very really still never always often "
    "sometimes maybe probably actually basically literally totally pretty quite "
    "need want feel love hate hope wish wait keep let put set try call ask tell "
    "talk speak hear listen watch move turn hold drop pick pull push bring carry "
    "show hide build create fix change update add remove install print import "
    "return true false none list dict string number value error debug line text "
    "name user app web link page site post video photo image music game food "
    "water coffee lunch dinner friend family mom dad brother sister dog cat car "
    "house room desk chair phone laptop computer camera light dark fast slow "
    "quick brown fox jumps lazy big small long short high low hot cold warm "
    "happy sad tired busy ready free done best better worse worst right wrong "
    "same different easy hard simple clean clear real life thing things place "
    "part point problem idea plan question answer fact story book paper pen "
    "money price cost buy sell pay order drive walk live stay leave arrive "
    "meet join follow share like comment watch learn teach remember forget "
    "understand believe mean seem become happen begin finish break win lose "
    "around through between under above before during without within against "
    "every each both few several much another own such while since until though "
    "although however therefore instead already almost enough together away "
    "again soon later early late tonight weekend monday tuesday wednesday "
    "thursday friday saturday sunday january february march april may june july "
    "august september october november december zero three four five six seven "
    "eight nine ten hundred thousand million"
).split()


_N_SAMPLES = 24
_UNIT_T = np.linspace(0.0, 1.0, _N_SAMPLES, dtype=np.float32)


def _path_length(points):
    arr = np.asarray(points, dtype=np.float32)
    if len(arr) < 2:
        return 0.0
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def _resample_path(points, n_samples=_N_SAMPLES):
    """Resamples a 2D polyline to `n_samples` evenly spaced points along arc length."""
    arr = np.asarray(points, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros((n_samples, 2), dtype=np.float32)
    if len(arr) == 1:
        return np.repeat(arr, n_samples, axis=0)

    diffs = np.diff(arr, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    total_len = float(np.sum(seg_lens))
    if total_len < 1e-6:
        return np.repeat(arr[:1], n_samples, axis=0)

    cum_dist = np.empty(len(arr), dtype=np.float32)
    cum_dist[0] = 0.0
    np.cumsum(seg_lens, out=cum_dist[1:])

    t_grid = _UNIT_T if n_samples == _N_SAMPLES else np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    target_dists = t_grid * total_len
    xs = np.interp(target_dists, cum_dist, arr[:, 0])
    ys = np.interp(target_dists, cum_dist, arr[:, 1])
    return np.column_stack((xs, ys)).astype(np.float32)


def _normalize_shape(resampled):
    """Zero-means and unit-scales a resampled path for shape comparison."""
    centered = resampled - np.mean(resampled, axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if scale < 1e-5:
        return centered
    return centered / scale


class _SwipeDecoder:
    """
    Offline geometric swipe-to-text decoder & fast frequency-ranked prefix autocomplete.
    """
    N_SAMPLES = _N_SAMPLES

    def __init__(self, key_unit_centers, wordlist_path="wordlist.txt"):
        self.key_centers = key_unit_centers
        self.words = self._load_words(wordlist_path)
        self.index = {}
        self.prefix_1 = {}
        self.prefix_2 = {}
        self.prefix_3 = {}
        self._build_templates()

    def _load_words(self, wordlist_path):
        seen = set()
        words = []

        candidates = []
        if os.path.exists(wordlist_path):
            try:
                with open(wordlist_path, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip().lower()
                        if w:
                            candidates.append(w)
                        if len(candidates) >= 25000:
                            break
            except OSError:
                candidates = []

        if not candidates:
            candidates = STARTER_WORDS

        for w in candidates:
            if len(w) >= 2 and w.isalpha() and w not in seen and all(ch in self.key_centers for ch in w):
                seen.add(w)
                words.append(w)
        return words

    def _build_templates(self):
        total = max(len(self.words), 1)
        key_centers = self.key_centers

        for rank, word in enumerate(self.words):
            # Prefix lookup buckets (words are already ordered by frequency)
            p1 = word[0]
            b1 = self.prefix_1.setdefault(p1, [])
            if len(b1) < 5:
                b1.append(word)

            p2 = word[:2]
            b2 = self.prefix_2.setdefault(p2, [])
            if len(b2) < 5:
                b2.append(word)

            if len(word) >= 3:
                self.prefix_3.setdefault(word[:3], []).append(word)

            # Swipe path template
            dedup = [word[0]]
            for ch in word[1:]:
                if ch != dedup[-1]:
                    dedup.append(ch)

            pts = np.array([key_centers[ch] for ch in dedup], dtype=np.float32)
            if len(pts) >= 2:
                diffs = np.diff(pts, axis=0)
                seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
                tpl_len = float(np.sum(seg_lens))
            else:
                tpl_len = 0.0

            loc = _resample_path(pts, self.N_SAMPLES)
            shape = _normalize_shape(loc)
            rank_penalty = 0.08 * (rank / total)
            self.index.setdefault((word[0], word[-1]), []).append(
                (word, loc, shape, tpl_len, pts, rank_penalty)
            )

    def suggest_prefix(self, prefix, top_k=5):
        """Returns up to `top_k` frequency-ranked words starting with `prefix`."""
        p = prefix.lower()
        if not p:
            return []
        if len(p) == 1:
            return self.prefix_1.get(p, [])[:top_k]
        if len(p) == 2:
            return self.prefix_2.get(p, [])[:top_k]

        bucket = self.prefix_3.get(p[:3], [])
        if len(p) == 3:
            return bucket[:top_k]

        matches = []
        for w in bucket:
            if w.startswith(p):
                matches.append(w)
                if len(matches) >= top_k:
                    break
        return matches

    def _nearby_keys(self, ux, uy, radius=1.25, min_count=3):
        dists = sorted(
            (math.hypot(ux - kx, uy - ky), ch)
            for ch, (kx, ky) in self.key_centers.items()
        )
        picked = {ch for d, ch in dists if d <= radius}
        for _, ch in dists[:min_count]:
            picked.add(ch)
        return picked

    def decode(self, unit_points, top_k=5):
        """Returns up to `top_k` best matching words for a swipe in key-unit coordinates."""
        if len(unit_points) < 2:
            return []

        user_len = _path_length(unit_points)
        if user_len < 0.8:
            return []

        user_loc = _resample_path(unit_points, self.N_SAMPLES)
        user_shape = _normalize_shape(user_loc)
        sx, sy = unit_points[0]
        ex, ey = unit_points[-1]

        start_keys = self._nearby_keys(sx, sy, radius=1.25, min_count=3)
        end_keys = self._nearby_keys(ex, ey, radius=1.25, min_count=3)

        scored = []
        for s_ch in start_keys:
            for e_ch in end_keys:
                bucket = self.index.get((s_ch, e_ch))
                if not bucket:
                    continue
                ksx, ksy = self.key_centers[s_ch]
                kex, key = self.key_centers[e_ch]
                endpoint_err = math.hypot(sx - ksx, sy - ksy) + math.hypot(ex - kex, ey - key)

                for word, tpl_loc, tpl_shape, tpl_len, uniq_pts, rank_penalty in bucket:
                    loc_err = float(np.mean(np.hypot(user_loc[:, 0] - tpl_loc[:, 0], user_loc[:, 1] - tpl_loc[:, 1])))
                    shape_err = float(np.mean(np.hypot(user_shape[:, 0] - tpl_shape[:, 0], user_shape[:, 1] - tpl_shape[:, 1])))
                    len_err = abs(user_len - tpl_len) / max(tpl_len, 1.0)

                    wp_dists = np.hypot(
                        uniq_pts[:, None, 0] - user_loc[None, :, 0],
                        uniq_pts[:, None, 1] - user_loc[None, :, 1],
                    )
                    min_per_letter = np.min(wp_dists, axis=1)
                    max_wp_err = float(np.max(min_per_letter))
                    mean_wp_err = float(np.mean(min_per_letter))

                    score = (
                        (0.55 * loc_err)
                        + (0.35 * shape_err)
                        + (0.30 * endpoint_err)
                        + (0.90 * len_err)
                        + (0.70 * max_wp_err)
                        + (0.45 * mean_wp_err)
                        + rank_penalty
                    )
                    scored.append((score, word))

        if not scored:
            return []

        scored.sort(key=lambda item: item[0])
        if scored[0][0] > 4.2:
            return []
        return [w for _, w in scored[:top_k]]


class HandKeyboard:
    """
    Focus-safe, click-through Smart Auto-Docking Slab keyboard with mobile
    tap + hold-and-swipe typing and a 5-slot clickable word suggestion bar.
    """

    # Colors
    BG_COLOR = "#12151C"
    BORDER_COLOR = "#2D3646"
    HEADER_BG = "#181C25"
    KEY_BG = "#1E2430"
    KEY_SPEC_BG = "#262E3D"
    KEY_HOVER_BG = "#2D3B52"
    KEY_ACTIVE_BG = "#00B8D4"
    KEY_SHIFT_ON_BG = "#1E465A"
    SUGG_BG = "#1D2432"
    SUGG_ACTIVE_BG = "#173A4A"
    SUGG_ACTIVE_OUTLINE = "#00E5FF"
    SUGG_OUTLINE = "#2E3A4F"
    KEY_TEXT = "#F1F5F9"
    KEY_SUBTEXT = "#94A3B8"
    SWIPE_TRAIL_COLOR = "#00E5FF"

    SUGGESTION_COUNT = 5

    # Interaction & Swipe Timing/Distance Tuning
    SWIPE_START_PX = 36.0
    MIN_SWIPE_UNITS = 1.15
    SWIPE_RELEASE_GRACE_SEC = 0.12
    SWIPE_RELEASE_TRIM_SEC = 0.06
    SHIFT_DOUBLE_TAP_SEC = 0.40
    BKSP_REPEAT_DELAY = 0.45
    BKSP_REPEAT_INTERVAL = 0.085
    CLOSE_COOLDOWN_SEC = 0.50

    ALPHA_ROWS = [
        [("q", 1.0), ("w", 1.0), ("e", 1.0), ("r", 1.0), ("t", 1.0), ("y", 1.0), ("u", 1.0), ("i", 1.0), ("o", 1.0), ("p", 1.0)],
        [("a", 1.0), ("s", 1.0), ("d", 1.0), ("f", 1.0), ("g", 1.0), ("h", 1.0), ("j", 1.0), ("k", 1.0), ("l", 1.0)],
        [("SHIFT", 1.45), ("z", 1.0), ("x", 1.0), ("c", 1.0), ("v", 1.0), ("b", 1.0), ("n", 1.0), ("m", 1.0), ("BKSP", 1.55)],
        [("123", 1.45), (",", 0.9), ("SPACE", 4.7), (".", 0.9), ("ENTER", 1.85)],
    ]

    SYMBOL_ROWS = [
        [("1", 1.0), ("2", 1.0), ("3", 1.0), ("4", 1.0), ("5", 1.0), ("6", 1.0), ("7", 1.0), ("8", 1.0), ("9", 1.0), ("0", 1.0)],
        [("@", 1.0), ("#", 1.0), ("$", 1.0), ("%", 1.0), ("&", 1.0), ("-", 1.0), ("+", 1.0), ("(", 1.0), (")", 1.0)],
        [("ABC", 1.45), ("!", 1.0), ("\"", 1.0), ("'", 1.0), (":", 1.0), (";", 1.0), ("/", 1.0), ("?", 1.0), ("BKSP", 1.55)],
        [("ABC", 1.45), (",", 0.9), ("SPACE", 4.7), (".", 0.9), ("ENTER", 1.85)],
    ]

    def __init__(self, master_root, wordlist_path="wordlist.txt"):
        self.root = master_root
        self.kb = KeyController()
        self.user32 = ctypes.windll.user32

        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()

        # Smart Auto-Docking Slab dimensions (~70% width, ~35% height; 48px suggestion header)
        self.width = max(720, min(int(self.screen_w * 0.70), 1440))
        self.height = max(258, min(int(self.screen_h * 0.35), 408))
        self.header_h = 48
        self.pad = 10
        self.key_gap = 6

        self.dock = "bottom"
        self.x = (self.screen_w - self.width) // 2
        self.y = self.screen_h - self.height - 56
        self.is_visible = False
        self.closed_at = 0.0

        # Keyboard & Word Suggestion State
        self.layer = "alpha"       # "alpha" or "symbol"
        self.shift_mode = "off"    # "off", "once", "caps"
        self.last_shift_tap = 0.0

        self.typed_buffer = ""             # Current prefix while tap-typing
        self.last_swiped_word = ""         # Most recent swiped word (without leading space)
        self.swipe_prefix_space = ""       # " " if auto-space was prepended, else ""
        self.last_swiped_len = 0           # Total chars typed for Smart Backspace undo
        self.auto_space_pending = False

        self.suggestion_mode = None        # None, "swipe", or "prefix"
        self.suggestions = []              # Formatted strings shown in slots 0..4
        self.active_suggestion_idx = None  # Highlighted slot index (for swipe replacement)

        # Pointer & Swipe State
        self.was_pressed = False
        self.press_key = None
        self.press_time = 0.0
        self.press_pos = (0.0, 0.0)
        self.hovered_key = None
        self.is_swiping = False
        self.swipe_points = []             # List of (timestamp, lx, ly)
        self.swipe_release_time = None
        self.swipe_trail_id = None
        self.bksp_next_repeat = 0.0
        self.bksp_did_repeat = False
        self.status_text = "TAP OR FIST/THUMB SWIPE"
        self.status_reset_at = 0.0

        # Build Toplevel Overlay Window
        self.win = tk.Toplevel(self.root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", 0.94)
        self.win.config(bg=self.BG_COLOR)

        self.canvas = tk.Canvas(
            self.win,
            width=self.width,
            height=self.height,
            bg=self.BG_COLOR,
            highlightthickness=0
        )
        self.canvas.pack()

        self.win.geometry(f"{self.width}x{self.height}+{self.x}+{self.y}")
        self.win.update_idletasks()
        self.hwnd = self.user32.GetParent(self.win.winfo_id()) or self.win.winfo_id()
        self._apply_clickthrough()

        # Precompute key geometry and swipe templates
        self.keys = []
        self.key_map = {}
        self.alpha_unit_centers = {}
        self.unit_origin_x = 0.0
        self.unit_origin_y = 0.0
        self.unit_w = 1.0
        self.unit_h = 1.0

        self._build_layout()
        self.decoder = _SwipeDecoder(self.alpha_unit_centers, wordlist_path=wordlist_path)

        # Start hidden without stealing focus
        self.user32.ShowWindow(self.hwnd, 0)  # SW_HIDE

    def _apply_clickthrough(self):
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_NOACTIVATE = 0x08000000

        style = self.user32.GetWindowLongW(self.hwnd, GWL_EXSTYLE)
        self.user32.SetWindowLongW(
            self.hwnd,
            GWL_EXSTYLE,
            style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        )

    def _compute_dock_position(self, dock):
        x = (self.screen_w - self.width) // 2
        if dock == "top":
            y = 22
        else:
            y = max(22, self.screen_h - self.height - 56)
        return x, y

    def _apply_geometry(self):
        self.x, self.y = self._compute_dock_position(self.dock)
        self.win.geometry(f"{self.width}x{self.height}+{self.x}+{self.y}")

    def show(self, cursor_y=None):
        if self.is_visible:
            return
        if (time.time() - self.closed_at) < self.CLOSE_COOLDOWN_SEC:
            return

        if cursor_y is not None:
            self.dock = "top" if cursor_y > (self.screen_h * 0.62) else "bottom"

        self._apply_geometry()
        self._update_dock_button_label()
        self.is_visible = True
        self._reset_interaction()

        SW_SHOWNOACTIVATE = 4
        HWND_TOPMOST = -1
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOACTIVATE = 0x0010
        self.user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
        self.user32.SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)

    def hide(self):
        if not self.is_visible:
            return
        self.is_visible = False
        self.closed_at = time.time()
        self._reset_interaction()
        self.auto_space_pending = False
        self.last_swiped_len = 0
        self.last_swiped_word = ""
        self.swipe_prefix_space = ""
        self.typed_buffer = ""
        self._clear_suggestions()
        self.user32.ShowWindow(self.hwnd, 0)  # SW_HIDE

    def toggle(self, cursor_y=None):
        if self.is_visible:
            self.hide()
        else:
            self.closed_at = 0.0
            self.show(cursor_y=cursor_y)

    def flip_dock(self):
        self.dock = "top" if self.dock == "bottom" else "bottom"
        self._apply_geometry()
        self._update_dock_button_label()

    def get_bounds(self):
        if not self.is_visible:
            return None
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def _reset_interaction(self):
        self.was_pressed = False
        self.press_key = None
        self.is_swiping = False
        self.swipe_release_time = None
        self.swipe_points.clear()
        self._clear_swipe_trail()
        self._set_hover(None)

    def _build_layout(self):
        self.canvas.delete("all")
        self.keys.clear()
        self.key_map.clear()
        self.hovered_key = None

        # Outer frame & header background
        self.canvas.create_rectangle(
            1, 1, self.width - 2, self.height - 2,
            fill=self.BG_COLOR, outline=self.BORDER_COLOR, width=2
        )
        self.canvas.create_rectangle(
            2, 2, self.width - 2, self.header_h,
            fill=self.HEADER_BG, outline=""
        )
        self.canvas.create_line(
            2, self.header_h, self.width - 2, self.header_h,
            fill=self.BORDER_COLOR, width=1
        )

        # Header Utility Buttons on right: [TOP/BOT] and [X]
        btn_y1 = 6
        btn_y2 = self.header_h - 6
        close_w = 46
        dock_w = 56
        close_x2 = self.width - 8
        close_x1 = close_x2 - close_w
        dock_x2 = close_x1 - 6
        dock_x1 = dock_x2 - dock_w

        self._add_key_rect("DOCK", "TOP" if self.dock == "bottom" else "BOT", dock_x1, btn_y1, dock_x2, btn_y2, is_special=True, font_size=8)
        self._add_key_rect("CLOSE", "X", close_x1, btn_y1, close_x2, btn_y2, is_special=True, font_size=10)

        # 5 Clickable Word Suggestion Pills across the remaining header width
        sugg_left = 8
        sugg_right = dock_x1 - 10
        sugg_gap = 6
        sugg_w = (sugg_right - sugg_left - sugg_gap * (self.SUGGESTION_COUNT - 1)) / self.SUGGESTION_COUNT

        for i in range(self.SUGGESTION_COUNT):
            sx1 = sugg_left + i * (sugg_w + sugg_gap)
            sx2 = sx1 + sugg_w
            self._add_key_rect(
                f"SUGG_{i}", "", sx1, btn_y1, sx2, btn_y2,
                is_special=False, font_size=10,
                custom_bg=self.SUGG_BG, custom_outline=self.SUGG_OUTLINE,
                enabled=False
            )

        # Idle/feedback hint (only visible when no word suggestions are active)
        self.status_item = self.canvas.create_text(
            (sugg_left + sugg_right) * 0.5, self.header_h // 2,
            text=self.status_text,
            fill=self.KEY_SUBTEXT,
            anchor="center",
            font=("Segoe UI", 9, "bold")
        )

        # Key Rows Area
        rows = self.ALPHA_ROWS if self.layer == "alpha" else self.SYMBOL_ROWS
        grid_top = self.header_h + self.pad
        grid_bottom = self.height - self.pad
        grid_left = self.pad
        grid_right = self.width - self.pad
        grid_w = grid_right - grid_left
        grid_h = grid_bottom - grid_top

        n_rows = len(rows)
        row_h = (grid_h - self.key_gap * (n_rows - 1)) / n_rows

        base_key_w = (grid_w - self.key_gap * 9) / 10.0
        if self.layer == "alpha" and not self.alpha_unit_centers:
            self.unit_origin_x = grid_left
            self.unit_origin_y = grid_top
            self.unit_w = base_key_w + self.key_gap
            self.unit_h = row_h + self.key_gap

        for r_idx, row in enumerate(rows):
            y1 = grid_top + r_idx * (row_h + self.key_gap)
            y2 = y1 + row_h

            total_weight = sum(w for _, w in row)
            row_total_w = total_weight * base_key_w + (len(row) - 1) * self.key_gap
            x_cursor = grid_left + max(0.0, (grid_w - row_total_w) / 2.0)

            for key_id, weight in row:
                kw = weight * base_key_w
                x1 = x_cursor
                x2 = x_cursor + kw
                x_cursor = x2 + self.key_gap

                is_special = key_id in ("SHIFT", "BKSP", "123", "ABC", "SPACE", "ENTER")
                display = self._display_label(key_id)
                font_size = 10 if is_special else 13
                self._add_key_rect(key_id, display, x1, y1, x2, y2, is_special=is_special, font_size=font_size)

                if self.layer == "alpha" and len(key_id) == 1 and key_id.isalpha():
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                    ux = (cx - self.unit_origin_x) / self.unit_w
                    uy = (cy - self.unit_origin_y) / self.unit_h
                    self.alpha_unit_centers[key_id] = (ux, uy)

        self._refresh_shift_visuals()
        self._refresh_suggestions_ui()

    def _add_key_rect(
        self, key_id, label, x1, y1, x2, y2,
        is_special=False, font_size=12,
        custom_bg=None, custom_outline="#2F384A", enabled=True
    ):
        bg = custom_bg if custom_bg is not None else (self.KEY_SPEC_BG if is_special else self.KEY_BG)
        state = "normal" if enabled else "hidden"
        rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill=bg, outline=custom_outline, width=1, state=state
        )
        txt_id = self.canvas.create_text(
            (x1 + x2) * 0.5, (y1 + y2) * 0.5,
            text=label, fill=self.KEY_TEXT,
            font=("Segoe UI", font_size, "bold"), state=state
        )
        key_obj = {
            "id": key_id,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) * 0.5, "cy": (y1 + y2) * 0.5,
            "is_special": is_special,
            "base_bg": bg,
            "base_outline": custom_outline,
            "enabled": enabled,
            "rect_id": rect_id,
            "txt_id": txt_id,
        }
        self.keys.append(key_obj)
        self.key_map[key_id] = key_obj

    def _display_label(self, key_id):
        if key_id == "SPACE":
            return "SPACE"
        if key_id == "BKSP":
            return "DEL"
        if key_id == "SHIFT":
            return "CAPS" if self.shift_mode == "caps" else "SHIFT"
        if len(key_id) == 1 and key_id.isalpha():
            return key_id.upper() if self.shift_mode != "off" else key_id
        return key_id

    def _refresh_shift_visuals(self):
        for k in self.keys:
            kid = k["id"]
            if len(kid) == 1 and kid.isalpha():
                self.canvas.itemconfig(k["txt_id"], text=self._display_label(kid))
            elif kid == "SHIFT":
                k["base_bg"] = self.KEY_SHIFT_ON_BG if self.shift_mode != "off" else self.KEY_SPEC_BG
                self.canvas.itemconfig(k["txt_id"], text=self._display_label(kid))
                if self.hovered_key != "SHIFT":
                    self.canvas.itemconfig(k["rect_id"], fill=k["base_bg"])

    def _update_dock_button_label(self):
        dock_key = self.key_map.get("DOCK")
        if dock_key:
            self.canvas.itemconfig(dock_key["txt_id"], text="TOP" if self.dock == "bottom" else "BOT")

    def _set_status(self, text, now=None, duration=0.0):
        self.status_text = text
        self.canvas.itemconfig(self.status_item, text=text)
        self.status_reset_at = (now + duration) if (now and duration > 0) else 0.0

    def _set_suggestions(self, words, mode, active_idx=None):
        self.suggestions = list(words[:self.SUGGESTION_COUNT])
        self.suggestion_mode = mode if self.suggestions else None
        self.active_suggestion_idx = active_idx if self.suggestions else None
        self._refresh_suggestions_ui()

    def _clear_suggestions(self):
        self.suggestions = []
        self.suggestion_mode = None
        self.active_suggestion_idx = None
        self._refresh_suggestions_ui()

    def _refresh_suggestions_ui(self):
        has_suggestions = bool(self.suggestions)
        self.canvas.itemconfig(self.status_item, state="hidden" if has_suggestions else "normal")

        for i in range(self.SUGGESTION_COUNT):
            kid = f"SUGG_{i}"
            slot = self.key_map.get(kid)
            if not slot:
                continue

            if i < len(self.suggestions):
                word = self.suggestions[i]
                is_active = (self.suggestion_mode == "swipe" and i == self.active_suggestion_idx)
                base_bg = self.SUGG_ACTIVE_BG if is_active else self.SUGG_BG
                base_outline = self.SUGG_ACTIVE_OUTLINE if is_active else self.SUGG_OUTLINE
                txt_color = "#67E8F9" if is_active else self.KEY_TEXT
                font_size = 9 if len(word) > 12 else 10

                slot["enabled"] = True
                slot["base_bg"] = base_bg
                slot["base_outline"] = base_outline

                fill = self.KEY_HOVER_BG if self.hovered_key == kid else base_bg
                outline = "#38BDF8" if self.hovered_key == kid else base_outline
                self.canvas.itemconfig(slot["rect_id"], fill=fill, outline=outline, state="normal")
                self.canvas.itemconfig(
                    slot["txt_id"],
                    text=word,
                    fill=txt_color,
                    font=("Segoe UI", font_size, "bold"),
                    state="normal"
                )
            else:
                slot["enabled"] = False
                if self.hovered_key == kid:
                    self.hovered_key = None
                self.canvas.itemconfig(slot["rect_id"], state="hidden")
                self.canvas.itemconfig(slot["txt_id"], text="", state="hidden")

    def _hit_test(self, lx, ly):
        for k in self.keys:
            if k["enabled"] and k["x1"] <= lx <= k["x2"] and k["y1"] <= ly <= k["y2"]:
                return k["id"]
        return None

    def _set_hover(self, key_id):
        if key_id == self.hovered_key:
            return
        if self.hovered_key and self.hovered_key in self.key_map:
            prev = self.key_map[self.hovered_key]
            if prev["enabled"]:
                self.canvas.itemconfig(prev["rect_id"], fill=prev["base_bg"], outline=prev["base_outline"])
        self.hovered_key = key_id
        if key_id and key_id in self.key_map:
            curr = self.key_map[key_id]
            fill = self.KEY_ACTIVE_BG if self.was_pressed else self.KEY_HOVER_BG
            self.canvas.itemconfig(curr["rect_id"], fill=fill, outline="#38BDF8")

    def _clear_swipe_trail(self):
        if self.swipe_trail_id is not None:
            self.canvas.delete(self.swipe_trail_id)
            self.swipe_trail_id = None

    def _draw_swipe_trail(self):
        self._clear_swipe_trail()
        if len(self.swipe_points) >= 2:
            flat = [coord for (_, x, y) in self.swipe_points for coord in (x, y)]
            self.swipe_trail_id = self.canvas.create_line(
                *flat,
                fill=self.SWIPE_TRAIL_COLOR,
                width=5,
                smooth=True,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND
            )

    @staticmethod
    def _format_word(word, casing):
        if casing == "caps":
            return word.upper()
        if casing == "title":
            return word.capitalize()
        return word

    def _consume_shift_casing(self):
        if self.shift_mode == "caps":
            return "caps"
        if self.shift_mode == "once":
            self.shift_mode = "off"
            self._refresh_shift_visuals()
            return "title"
        return "lower"

    def _buffer_casing(self):
        if self.shift_mode == "caps" or (len(self.typed_buffer) > 1 and self.typed_buffer.isupper()):
            return "caps"
        if self.typed_buffer and self.typed_buffer[0].isupper():
            return "title"
        return "lower"

    def _update_prefix_suggestions(self):
        if not self.typed_buffer:
            self._clear_suggestions()
            return
        raw = self.decoder.suggest_prefix(self.typed_buffer, top_k=self.SUGGESTION_COUNT)
        casing = self._buffer_casing()
        formatted = [self._format_word(w, casing) for w in raw]
        self._set_suggestions(formatted, mode="prefix", active_idx=None)

    def _apply_suggestion(self, idx):
        if idx >= len(self.suggestions):
            return
        chosen = self.suggestions[idx]

        if self.suggestion_mode == "swipe":
            if idx == self.active_suggestion_idx:
                return
            # Replace only the swiped word, preserving any leading auto-space
            for _ in range(len(self.last_swiped_word)):
                self.kb.tap(Key.backspace)
            self.kb.type(chosen)
            self.last_swiped_word = chosen
            self.last_swiped_len = len(self.swipe_prefix_space) + len(chosen)
            self.active_suggestion_idx = idx
            self.auto_space_pending = True
            self._refresh_suggestions_ui()

        elif self.suggestion_mode == "prefix":
            # Append only the remaining suffix + trailing space (no backspace needed)
            suffix = chosen[len(self.typed_buffer):]
            self.kb.type(suffix + " ")
            self.typed_buffer = ""
            self.last_swiped_len = 0
            self.last_swiped_word = ""
            self.swipe_prefix_space = ""
            self.auto_space_pending = False
            self._clear_suggestions()

    def _execute_tap(self, key_id, now):
        if key_id.startswith("SUGG_"):
            self._apply_suggestion(int(key_id.split("_")[1]))
            return
        if key_id == "CLOSE":
            self.hide()
            return
        if key_id == "DOCK":
            self.flip_dock()
            return
        if key_id == "123":
            self.layer = "symbol"
            self._build_layout()
            return
        if key_id == "ABC":
            self.layer = "alpha"
            self._build_layout()
            return
        if key_id == "SHIFT":
            if self.shift_mode == "off":
                self.shift_mode = "once"
            elif self.shift_mode == "once":
                self.shift_mode = "caps" if (now - self.last_shift_tap) <= self.SHIFT_DOUBLE_TAP_SEC else "off"
            else:
                self.shift_mode = "off"
            self.last_shift_tap = now
            self._refresh_shift_visuals()
            return

        if key_id == "BKSP":
            if self.last_swiped_len > 0:
                for _ in range(self.last_swiped_len):
                    self.kb.tap(Key.backspace)
                self.last_swiped_len = 0
                self.last_swiped_word = ""
                self.swipe_prefix_space = ""
                self.auto_space_pending = False
                self._clear_suggestions()
                self._set_status("UNDID SWIPE", now, 0.9)
            else:
                self.kb.tap(Key.backspace)
                self.auto_space_pending = False
                if self.typed_buffer:
                    self.typed_buffer = self.typed_buffer[:-1]
                    self._update_prefix_suggestions()
                else:
                    self._clear_suggestions()
            return

        if key_id == "SPACE":
            self.kb.tap(Key.space)
            self.typed_buffer = ""
            self.last_swiped_len = 0
            self.last_swiped_word = ""
            self.swipe_prefix_space = ""
            self.auto_space_pending = False
            self._clear_suggestions()
            return

        if key_id == "ENTER":
            self.kb.tap(Key.enter)
            self.typed_buffer = ""
            self.last_swiped_len = 0
            self.last_swiped_word = ""
            self.swipe_prefix_space = ""
            self.auto_space_pending = False
            self._clear_suggestions()
            return

        # Standard letter or punctuation character
        self.last_swiped_len = 0
        self.last_swiped_word = ""
        self.swipe_prefix_space = ""
        if len(key_id) == 1 and key_id.isalpha():
            ch = key_id.upper() if self.shift_mode != "off" else key_id
            if self.shift_mode == "once":
                self.shift_mode = "off"
                self._refresh_shift_visuals()
            self.kb.type(ch)
            self.auto_space_pending = False
            self.typed_buffer += ch
            self._update_prefix_suggestions()
        else:
            self.kb.type(key_id)
            self.auto_space_pending = False
            self.typed_buffer = ""
            self._clear_suggestions()

    def _finish_swipe(self, now):
        pts = self.swipe_points
        if len(pts) > 3:
            end_t = pts[-1][0]
            trimmed = [p for p in pts if (end_t - p[0]) >= self.SWIPE_RELEASE_TRIM_SEC]
            if len(trimmed) >= 2:
                pts = trimmed

        unit_pts = [
            ((lx - self.unit_origin_x) / self.unit_w, (ly - self.unit_origin_y) / self.unit_h)
            for (_, lx, ly) in pts
        ]

        if _path_length(unit_pts) < self.MIN_SWIPE_UNITS:
            if self.press_key:
                self._execute_tap(self.press_key, now)
            return

        matches = self.decoder.decode(unit_pts, top_k=self.SUGGESTION_COUNT)
        if not matches:
            self._clear_suggestions()
            self._set_status("NO MATCH", now, 0.8)
            return

        casing = self._consume_shift_casing()
        formatted = [self._format_word(w, casing) for w in matches]
        word = formatted[0]

        prefix = " " if (self.auto_space_pending or self.typed_buffer) else ""
        out_text = prefix + word
        self.kb.type(out_text)

        self.typed_buffer = ""
        self.swipe_prefix_space = prefix
        self.last_swiped_word = word
        self.last_swiped_len = len(out_text)
        self.auto_space_pending = True
        self._set_suggestions(formatted, mode="swipe", active_idx=0)

    def update(self, cursor_x, cursor_y, is_pressed, now):
        """Processes cursor position and virtual press state (thumb, fist, or index drop) for one frame."""
        if not self.is_visible:
            return

        if self.status_reset_at and now >= self.status_reset_at:
            self._set_status("TAP OR FIST/THUMB SWIPE")

        lx = cursor_x - self.x
        ly = cursor_y - self.y
        over_key = self._hit_test(lx, ly)
        self._set_hover(over_key)

        # If mid-swipe release grace is active and user re-engages press, resume seamlessly
        if is_pressed and self.swipe_release_time is not None:
            self.swipe_release_time = None

        # Press Down
        if is_pressed and not self.was_pressed:
            self.was_pressed = True
            self.press_key = over_key
            self.press_time = now
            self.press_pos = (lx, ly)
            self.is_swiping = False
            self.swipe_release_time = None
            self.swipe_points = [(now, lx, ly)]
            self.bksp_did_repeat = False
            self.bksp_next_repeat = now + self.BKSP_REPEAT_DELAY

            if over_key and over_key in self.key_map:
                self.canvas.itemconfig(self.key_map[over_key]["rect_id"], fill=self.KEY_ACTIVE_BG)

        # Holding
        elif is_pressed and self.was_pressed:
            if self.press_key == "BKSP" and over_key == "BKSP":
                if now >= self.bksp_next_repeat:
                    self.kb.tap(Key.backspace)
                    self.last_swiped_len = 0
                    self.last_swiped_word = ""
                    self.swipe_prefix_space = ""
                    self.auto_space_pending = False
                    if self.typed_buffer:
                        self.typed_buffer = self.typed_buffer[:-1]
                        self._update_prefix_suggestions()
                    else:
                        self._clear_suggestions()
                    self.bksp_did_repeat = True
                    self.bksp_next_repeat = now + self.BKSP_REPEAT_INTERVAL

            elif self.layer == "alpha" and self.press_key and len(self.press_key) == 1 and self.press_key.isalpha():
                clamped_x = max(self.pad, min(self.width - self.pad, lx))
                clamped_y = max(self.header_h + self.pad, min(self.height - self.pad, ly))

                _, last_x, last_y = self.swipe_points[-1]
                if math.hypot(clamped_x - last_x, clamped_y - last_y) >= 4.0:
                    self.swipe_points.append((now, clamped_x, clamped_y))

                if not self.is_swiping:
                    moved = math.hypot(clamped_x - self.press_pos[0], clamped_y - self.press_pos[1])
                    left_start_key = (
                        over_key is not None
                        and over_key != self.press_key
                        and len(over_key) == 1
                        and over_key.isalpha()
                    )
                    if moved >= self.SWIPE_START_PX or (left_start_key and moved >= self.SWIPE_START_PX * 0.75):
                        self.is_swiping = True

                if self.is_swiping:
                    self._draw_swipe_trail()

        # Release (immediate for taps; brief grace window for swipes to absorb flicker)
        elif not is_pressed and self.was_pressed:
            if self.is_swiping:
                if self.swipe_release_time is None:
                    self.swipe_release_time = now
                if (now - self.swipe_release_time) < self.SWIPE_RELEASE_GRACE_SEC:
                    return

                self.was_pressed = False
                self.swipe_release_time = None
                self._clear_swipe_trail()
                self._finish_swipe(now)
            else:
                self.was_pressed = False
                self._clear_swipe_trail()
                if not (self.press_key == "BKSP" and self.bksp_did_repeat):
                    target_key = over_key if over_key == self.press_key else self.press_key
                    if target_key:
                        self._execute_tap(target_key, now)

            self.press_key = None
            self.is_swiping = False
            self.swipe_points.clear()

            if self.hovered_key and self.hovered_key in self.key_map and self.key_map[self.hovered_key]["enabled"]:
                self.canvas.itemconfig(self.key_map[self.hovered_key]["rect_id"], fill=self.KEY_HOVER_BG)
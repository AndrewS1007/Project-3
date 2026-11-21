import ctypes
import tkinter as tk


class CursorIndicator:
    STATES = {
        "INACTIVE": ("INACTIVE", "#FFA000"),
        "SCROLL": ("SCROLL", "#FF3B30"),
        "CLICK": ("CLICK", "#00E676"),
        "DRAG": ("DRAG", "#00E676"),
        "2x CLICK": ("2x CLICK", "#00E5FF"),
        "RIGHT": ("RIGHT", "#FF40FF"),
        "RIGHT HAND": ("RIGHT HAND", "#40C4FF"),
        "LEFT HAND": ("LEFT HAND", "#40C4FF"),
    }

    def __init__(self, hud_corner="bottom-right"):
        self.transparent_color = "#010101"

        # 1. Floating Cursor Action Badge
        self.root = tk.Tk()
        self._configure_overlay_window(self.root)

        self.width = 130
        self.height = 30
        self.canvas = tk.Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg=self.transparent_color,
            highlightthickness=0
        )
        self.canvas.pack()

        self.bg_rect = self.canvas.create_rectangle(
            2, 2, self.width - 2, self.height - 2,
            fill="#181818", outline="#00E676", width=2, state="hidden"
        )
        self.dot = self.canvas.create_oval(
            10, 9, 22, 21,
            fill="#00E676", outline="", state="hidden"
        )
        self.label = self.canvas.create_text(
            74, 15, text="", fill="#FFFFFF",
            font=("Segoe UI", 9, "bold"), state="hidden"
        )
        self._apply_clickthrough(self.root)

        # 2. Persistent Corner HUD (Always shows active hand + Active/Paused dot)
        self.hud = tk.Toplevel(self.root)
        self._configure_overlay_window(self.hud)

        self.hud_width = 132
        self.hud_height = 30
        self.hud_canvas = tk.Canvas(
            self.hud,
            width=self.hud_width,
            height=self.hud_height,
            bg=self.transparent_color,
            highlightthickness=0
        )
        self.hud_canvas.pack()

        self.hud_border = self.hud_canvas.create_rectangle(
            2, 2, self.hud_width - 2, self.hud_height - 2,
            fill="#161616", outline="#3A3A3A", width=2
        )
        self.hud_status_dot = self.hud_canvas.create_oval(
            10, 10, 19, 19, fill="#00E676", outline=""
        )
        self.hud_left_bg = self.hud_canvas.create_rectangle(
            26, 5, 74, 25, fill="#161616", outline=""
        )
        self.hud_right_bg = self.hud_canvas.create_rectangle(
            77, 5, 126, 25, fill="#1E3A4C", outline=""
        )
        self.hud_left_txt = self.hud_canvas.create_text(
            50, 15, text="LEFT", fill="#666666", font=("Segoe UI", 8, "bold")
        )
        self.hud_right_txt = self.hud_canvas.create_text(
            101, 15, text="RIGHT", fill="#40C4FF", font=("Segoe UI", 8, "bold")
        )

        self._position_corner_hud(hud_corner)
        self._apply_clickthrough(self.hud)

        self.visible = False
        self.last_state = None
        self.last_pos = (None, None)
        self.last_hud_state = None

    def _configure_overlay_window(self, win):
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.config(bg=self.transparent_color)
        win.attributes("-transparentcolor", self.transparent_color)

    def _apply_clickthrough(self, win):
        win.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id()) or win.winfo_id()
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_NOACTIVATE = 0x08000000

        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(
            hwnd,
            GWL_EXSTYLE,
            style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        )

    def _position_corner_hud(self, corner):
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        pad_x = 18

        if corner == "top-right":
            x, y = screen_w - self.hud_width - pad_x, 16
        elif corner == "top-left":
            x, y = pad_x, 16
        elif corner == "bottom-left":
            x, y = pad_x, screen_h - self.hud_height - 58
        else:  # bottom-right (just above Windows taskbar)
            x, y = screen_w - self.hud_width - pad_x, screen_h - self.hud_height - 58

        self.hud.geometry(f"{self.hud_width}x{self.hud_height}+{x}+{y}")

    def _update_corner_hud(self, active_hand, is_active):
        hud_state = (active_hand, is_active)
        if hud_state == self.last_hud_state:
            return

        border_color = "#3A3A3A" if is_active else "#FFA000"
        dot_color = "#00E676" if is_active else "#FFA000"
        self.hud_canvas.itemconfig(self.hud_border, outline=border_color)
        self.hud_canvas.itemconfig(self.hud_status_dot, fill=dot_color)

        if active_hand == "Left":
            self.hud_canvas.itemconfig(self.hud_left_bg, fill="#1E3A4C")
            self.hud_canvas.itemconfig(self.hud_left_txt, fill="#40C4FF")
            self.hud_canvas.itemconfig(self.hud_right_bg, fill="#161616")
            self.hud_canvas.itemconfig(self.hud_right_txt, fill="#666666")
        else:
            self.hud_canvas.itemconfig(self.hud_left_bg, fill="#161616")
            self.hud_canvas.itemconfig(self.hud_left_txt, fill="#666666")
            self.hud_canvas.itemconfig(self.hud_right_bg, fill="#1E3A4C")
            self.hud_canvas.itemconfig(self.hud_right_txt, fill="#40C4FF")

        self.last_hud_state = hud_state

    def update(self, state=None, active_hand="Right", is_active=True):
        self._update_corner_hud(active_hand, is_active)
        config = self.STATES.get(state) if state else None

        if config:
            text, color = config
            px, py = self.root.winfo_pointerxy()
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()

            # Position centered below cursor; flip above if near bottom edge
            x = max(0, min(px - self.width // 2, screen_w - self.width))
            y = py + 24 if (py + 24 + self.height <= screen_h) else py - self.height - 12

            if (x, y) != self.last_pos:
                self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")
                self.last_pos = (x, y)

            if state != self.last_state or not self.visible:
                self.canvas.itemconfig(self.bg_rect, outline=color, state="normal")
                self.canvas.itemconfig(self.dot, fill=color, state="normal")
                self.canvas.itemconfig(self.label, text=text, state="normal")
                self.last_state = state
                self.visible = True
        elif self.visible:
            self.canvas.itemconfig(self.bg_rect, state="hidden")
            self.canvas.itemconfig(self.dot, state="hidden")
            self.canvas.itemconfig(self.label, state="hidden")
            self.visible = False
            self.last_state = None

        self.root.update_idletasks()
        self.root.update()

    def close(self):
        self.root.destroy()
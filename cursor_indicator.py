import ctypes
import tkinter as tk


class CursorIndicator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        # Near-black chroma key treated as 100% transparent by Windows
        self.transparent_color = "#010101"
        self.root.config(bg=self.transparent_color)
        self.root.attributes("-transparentcolor", self.transparent_color)

        self.width = 110
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
            64, 15, text="", fill="#FFFFFF",
            font=("Segoe UI", 9, "bold"), state="hidden"
        )

        # Apply Windows click-through, no-focus, and hide-from-Alt+Tab styles
        self.root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
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

        self.visible = False
        self.last_state = (None, None)
        self.last_pos = (None, None)

    def update(self, text=None, color=None):
        if text and color:
            px, py = self.root.winfo_pointerxy()
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()

            # Position centered below cursor; flip above if near bottom edge
            x = max(0, min(px - self.width // 2, screen_w - self.width))
            y = py + 24 if (py + 24 + self.height <= screen_h) else py - self.height - 12

            if (x, y) != self.last_pos:
                self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")
                self.last_pos = (x, y)

            if (text, color) != self.last_state or not self.visible:
                self.canvas.itemconfig(self.bg_rect, outline=color, state="normal")
                self.canvas.itemconfig(self.dot, fill=color, state="normal")
                self.canvas.itemconfig(self.label, text=text, state="normal")
                self.last_state = (text, color)
                self.visible = True
        elif self.visible:
            self.canvas.itemconfig(self.bg_rect, state="hidden")
            self.canvas.itemconfig(self.dot, state="hidden")
            self.canvas.itemconfig(self.label, state="hidden")
            self.visible = False
            self.last_state = (None, None)

        self.root.update_idletasks()
        self.root.update()

    def close(self):
        self.root.destroy()
"""Tk windows: presenter overlay, laser dot, settings, guided setup, gaze calibration, keyboard.

All windows are driven from the camera loop by calling TkUI.update() once per frame.
"""

import logging

from .config import FACE_GESTURES, save_json

log = logging.getLogger("slidecontrol")

BG = "#1e1f24"
PANEL = "#2a2c33"
FG = "#f2f2f2"
MUTED = "#9aa0a6"
ACCENT = "#4fc3f7"
GREEN = "#66bb6a"
RED = "#ef5350"
YELLOW = "#ffca28"


class TkUI:
    def __init__(self, app):
        import tkinter as tk
        self.tk = tk
        self.app = app
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Face Slide Control")
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.overlay = None
        self.laser = None
        self.settings = None
        self.setup = None
        self.gaze = None
        self.keyboard = None

    @classmethod
    def create(cls, app):
        try:
            return cls(app)
        except Exception as exc:
            log.warning("On-screen windows unavailable (%s); overlay, settings and keyboard disabled", exc)
            return None

    def update(self, now):
        try:
            for win in (self.overlay, self.laser, self.settings, self.setup, self.gaze, self.keyboard):
                if win is not None:
                    win.refresh(now)
            self.root.update()
        except self.tk.TclError as exc:
            log.debug("Tk error: %s", exc)

    def destroy(self):
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass

    # Opening/closing helpers -------------------------------------------------

    def toggle(self, attr, factory, show):
        win = getattr(self, attr)
        if show and win is None:
            setattr(self, attr, factory(self))
        elif not show and win is not None:
            win.close()
            setattr(self, attr, None)

    def set_overlay(self, show):
        self.toggle("overlay", PresenterOverlay, show)

    def set_laser(self, show):
        self.toggle("laser", LaserDot, show)

    def set_keyboard(self, show):
        self.toggle("keyboard", KeyboardWindow, show)

    def open_settings(self):
        if self.settings is None:
            self.settings = SettingsWindow(self)
        else:
            self.settings.win.lift()

    def show_setup(self, show):
        self.toggle("setup", SetupWindow, show)

    def show_gaze_calibration(self, show):
        self.toggle("gaze", GazeCalibrationWindow, show)


class Window:
    def __init__(self, ui, overrideredirect=False, topmost=True, alpha=None):
        self.ui = ui
        self.app = ui.app
        tk = ui.tk
        self.win = tk.Toplevel(ui.root, bg=BG)
        if overrideredirect:
            self.win.overrideredirect(True)
        if topmost:
            self.win.attributes("-topmost", True)
        if alpha is not None:
            try:
                self.win.attributes("-alpha", alpha)
            except tk.TclError:
                pass

    def close(self):
        try:
            self.win.destroy()
        except self.ui.tk.TclError:
            pass

    def refresh(self, now):
        pass

    def label(self, parent, text="", size=11, color=FG, bold=False, **kw):
        font = ("Sans", size, "bold" if bold else "normal")
        return self.ui.tk.Label(parent, text=text, fg=color, bg=kw.pop("bg", BG), font=font, **kw)


class PresenterOverlay(Window):
    """Small always-on-top panel: timer, status, profile, slide steps, last gesture. Drag to move."""

    def __init__(self, ui):
        super().__init__(ui, overrideredirect=True, alpha=0.88)
        w, h = 270, 128
        self.win.geometry(f"{w}x{h}+{ui.screen_w - w - 24}+{48}")
        self.timer = self.label(self.win, "00:00", 26, bold=True)
        self.timer.pack(anchor="w", padx=12, pady=(6, 0))
        self.status = self.label(self.win, "", 11, bold=True)
        self.status.pack(anchor="w", padx=12)
        self.info = self.label(self.win, "", 10, MUTED)
        self.info.pack(anchor="w", padx=12)
        self.last = self.label(self.win, "", 10, YELLOW)
        self.last.pack(anchor="w", padx=12)
        for widget in (self.win, self.timer, self.status, self.info, self.last):
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._drag)

    def _start_drag(self, event):
        self._dx, self._dy = event.x_root - self.win.winfo_x(), event.y_root - self.win.winfo_y()

    def _drag(self, event):
        self.win.geometry(f"+{event.x_root - self._dx}+{event.y_root - self._dy}")

    def refresh(self, now):
        app = self.app
        elapsed = int(now - app.start_time)
        self.timer.config(text=f"{elapsed // 3600:d}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d}"
                          if elapsed >= 3600 else f"{elapsed // 60:02d}:{elapsed % 60:02d}")
        status, color = app.status_text()
        self.status.config(text=status, fg=color)
        rate = app.wellness.blinks.rate(now)
        blink = f"  ·  {rate:.0f} blinks/min" if rate is not None else ""
        self.info.config(text=f"profile {app.cfg.profile or 'default'}  ·  steps {app.dispatcher.position:+d}{blink}")
        d = app.dispatcher
        self.last.config(text=d.last_action if d.last_action and now - d.last_action_time < 3 else "")


class LaserDot(Window):
    SIZE = 22

    def __init__(self, ui):
        super().__init__(ui, overrideredirect=True, alpha=0.85)
        self.win.config(bg=RED)
        self.win.geometry(f"{self.SIZE}x{self.SIZE}+0+0")
        self.win.withdraw()
        self.visible = False

    def refresh(self, now):
        point = self.app.hand.laser_point
        if point is None:
            if self.visible:
                self.win.withdraw()
                self.visible = False
            return
        x = int(point[0] * self.ui.screen_w) - self.SIZE // 2
        y = int(point[1] * self.ui.screen_h) - self.SIZE // 2
        self.win.geometry(f"+{x}+{y}")
        if not self.visible:
            self.win.deiconify()
            self.visible = True


class SetupWindow(Window):
    def __init__(self, ui):
        super().__init__(ui)
        self.win.title("Guided setup")
        w, h = 560, 190
        self.win.geometry(f"{w}x{h}+{(ui.screen_w - w) // 2}+{ui.screen_h // 6}")
        self.step = self.label(self.win, "", 10, MUTED)
        self.step.pack(pady=(14, 0))
        self.text = self.label(self.win, "", 17, bold=True, wraplength=520)
        self.text.pack(pady=10)
        self.canvas = ui.tk.Canvas(self.win, width=480, height=14, bg=PANEL, highlightthickness=0)
        self.canvas.pack()
        self.bar = self.canvas.create_rectangle(0, 0, 0, 14, fill=ACCENT, width=0)

    def refresh(self, now):
        setup = self.app.setup
        if setup is None or setup.finished:
            return
        _, instruction, _ = setup.current()
        self.step.config(text=f"Step {setup.index + 1} of {len(setup.steps)}")
        self.text.config(text=instruction)
        self.canvas.coords(self.bar, 0, 0, 480 * setup.progress(now), 14)


class GazeCalibrationWindow(Window):
    def __init__(self, ui):
        super().__init__(ui)
        self.win.config(bg="black")
        self.win.attributes("-fullscreen", True)
        self.canvas = ui.tk.Canvas(self.win, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.win.bind("<Escape>", lambda e: self.app.cancel_gaze_calibration())

    def refresh(self, now):
        cal = self.app.gaze_calibration
        self.canvas.delete("all")
        w, h = self.ui.screen_w, self.ui.screen_h
        self.canvas.create_text(w // 2, h - 40, fill=MUTED, font=("Sans", 14),
                                text="Keep your head still and look at each dot until it turns red.  Esc cancels.")
        state = cal.current(now) if cal else None
        if state is None:
            return
        index, (px, py), sampling = state
        x, y = px * w, py * h
        r = 14 if sampling else 22
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=RED if sampling else "white", outline="")
        self.canvas.create_text(w // 2, 40, fill=MUTED, font=("Sans", 12), text=f"{index + 1} / {len(cal.points)}")


class KeyboardWindow(Window):
    """Scanning keyboard: wink left = select highlighted row/key, wink right = back.
    Keys can also be clicked with the hand or gaze pointer."""

    def __init__(self, ui):
        super().__init__(ui, overrideredirect=True, alpha=0.95)
        kb = self.app.keyboard
        self.cells = []
        self.typed = self.label(self.win, "", 13, ACCENT, anchor="w")
        self.typed.pack(fill="x", padx=10, pady=(8, 2))
        grid = ui.tk.Frame(self.win, bg=BG)
        grid.pack(padx=8, pady=(0, 8))
        for r, row in enumerate(kb.layout):
            cells = []
            for c, key in enumerate(row):
                cell = self.label(grid, key, 15, bold=True, bg=PANEL, width=6 if len(row) > 1 else 46, pady=8)
                cell.grid(row=r, column=c, columnspan=1 if len(row) > 1 else len(kb.layout[0]), padx=2, pady=2)
                cell.bind("<Button-1>", lambda e, k=key: self.app.keyboard_press(k))
                cells.append(cell)
            self.cells.append(cells)
        self.win.update_idletasks()
        w, h = self.win.winfo_reqwidth(), self.win.winfo_reqheight()
        self.win.geometry(f"+{(ui.screen_w - w) // 2}+{ui.screen_h - h - 60}")

    def refresh(self, now):
        kb = self.app.keyboard
        if kb is None:
            return
        for r, cells in enumerate(self.cells):
            for c, cell in enumerate(cells):
                if r == kb.row and (kb.col is None or kb.col == c):
                    color = YELLOW if kb.col is not None else ACCENT
                    cell.config(bg=color, fg=BG)
                else:
                    cell.config(bg=PANEL, fg=FG)
        self.typed.config(text="typed: " + kb.typed[-40:] + "▏")


SLIDERS = {
    "Face": [
        ("close_ratio", "Eye closed below (× open)", 0.2, 0.9, 0.01),
        ("open_ratio", "Other eye open above (× open)", 0.3, 1.0, 0.01),
        ("wink_hold", "Wink hold (s)", 0.05, 1.0, 0.05),
        ("long_blink_hold", "Long blink (s)", 0.5, 3.0, 0.1),
        ("brow_raise_ratio", "Brow raise (× normal)", 1.02, 1.6, 0.01),
        ("smile_ratio", "Smile width (× normal)", 1.02, 1.6, 0.01),
        ("mouth_open_threshold", "Mouth open above", 0.2, 0.9, 0.01),
        ("cooldown", "Cooldown between actions (s)", 0.0, 2.0, 0.05),
        ("repeat_delay", "Repeat starts after (s)", 0.1, 2.0, 0.05),
        ("repeat_interval", "Repeat every (s)", 0.04, 1.0, 0.02),
        ("head_scroll_deadzone", "Head scroll dead zone", 0.005, 0.1, 0.005),
        ("head_scroll_speed", "Head scroll speed", 2, 40, 1),
    ],
    "Hands & pointer": [
        ("pointer_speed", "Pointer speed", 500, 5000, 100),
        ("pointer_smoothing", "Pointer smoothing", 0.0, 0.95, 0.05),
        ("pinch_threshold", "Pinch closes below", 0.1, 0.5, 0.01),
        ("pinch_release", "Pinch opens above", 0.15, 0.8, 0.01),
        ("swipe_distance", "Swipe distance (× frame)", 0.1, 0.6, 0.01),
        ("hand_scroll_speed", "Two-finger scroll speed", 5, 120, 5),
        ("gaze_smoothing", "Gaze smoothing", 0.0, 0.97, 0.01),
    ],
}
TOGGLES = [
    ("hand_control", "Hand control"), ("head_scroll", "Head-tilt scrolling"), ("gaze", "Gaze pointer"),
    ("gaze_dwell_click", "Gaze dwell click"), ("voice", "Voice commands"), ("overlay", "Presenter overlay"),
    ("repeat_enabled", "Repeat while held"), ("auto_pause_away", "Pause when I look away"),
    ("pause_on_second_face", "Pause when someone else appears"), ("sounds", "Sounds"),
    ("notifications", "Notifications"), ("show_mesh", "Show face points"),
]


class SettingsWindow(Window):
    def __init__(self, ui):
        super().__init__(ui, topmost=False)
        tk = ui.tk
        from tkinter import ttk
        self.win.title("Face Slide Control – Settings")
        self.win.protocol("WM_DELETE_WINDOW", self.close_window)
        cfg = self.app.cfg
        self.vars = {}

        notebook = ttk.Notebook(self.win)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)
        for tab, sliders in SLIDERS.items():
            frame = tk.Frame(notebook, bg=BG)
            notebook.add(frame, text=tab)
            for i, (name, text, lo, hi, step) in enumerate(sliders):
                self.label(frame, text, 10).grid(row=i, column=0, sticky="w", padx=8)
                var = tk.DoubleVar(value=getattr(cfg, name))
                scale = tk.Scale(frame, variable=var, from_=lo, to=hi, resolution=step, orient="horizontal",
                                 length=260, bg=BG, fg=FG, highlightthickness=0, troughcolor=PANEL,
                                 command=lambda v, n=name: self.set_value(n, v))
                scale.grid(row=i, column=1, padx=8)
                self.vars[name] = var

        features = tk.Frame(notebook, bg=BG)
        notebook.add(features, text="Features")
        for i, (name, text) in enumerate(TOGGLES):
            var = tk.BooleanVar(value=getattr(cfg, name))
            tk.Checkbutton(features, text=text, variable=var, bg=BG, fg=FG, selectcolor=PANEL,
                           activebackground=BG, activeforeground=FG,
                           command=lambda n=name, v=var: self.set_value(n, v.get())).grid(
                row=i % 6, column=i // 6, sticky="w", padx=10, pady=3)
            self.vars[name] = var

        gestures = tk.Frame(notebook, bg=BG)
        notebook.add(gestures, text="Gestures")
        self.gesture_vars = {}
        for i, g in enumerate(FACE_GESTURES):
            var = tk.BooleanVar(value=g in cfg.gestures)
            tk.Checkbutton(gestures, text=f"{g}  →  {cfg.actions.get(g)}", variable=var, bg=BG, fg=FG,
                           selectcolor=PANEL, activebackground=BG, activeforeground=FG,
                           command=self.set_gestures).grid(row=i % 5, column=i // 5, sticky="w", padx=10, pady=3)
            self.gesture_vars[g] = var

        buttons = tk.Frame(self.win, bg=BG)
        buttons.pack(fill="x", padx=8, pady=(0, 8))
        for text, cmd in [("Save", self.save), ("Guided setup", self.app.start_setup),
                          ("Calibrate gaze", self.app.start_gaze_calibration),
                          ("Recalibrate face", self.app.recalibrate), ("Close", self.close_window)]:
            tk.Button(buttons, text=text, command=cmd, bg=PANEL, fg=FG, activebackground=ACCENT,
                      relief="flat", padx=10).pack(side="left", padx=4)
        self.saved = self.label(buttons, "", 10, GREEN)
        self.saved.pack(side="left", padx=8)

    def set_value(self, name, value):
        cfg = self.app.cfg
        old = getattr(cfg, name)
        setattr(cfg, name, type(old)(float(value)) if not isinstance(old, bool) else bool(value))
        try:
            cfg.validate()
        except ValueError as exc:
            setattr(cfg, name, old)
            self.saved.config(text=str(exc), fg=RED)
            return
        self.saved.config(text="")
        self.app.apply_config()

    def set_gestures(self):
        self.app.cfg.gestures = [g for g, v in self.gesture_vars.items() if v.get()]
        self.app.apply_config()

    def save(self):
        path = self.app.config_path
        # One-off command-line options (e.g. --dry-run, --no-preview) are not made permanent.
        data = {k: v for k, v in self.app.cfg.to_dict().items()
                if k not in self.app.cli_data and k != "dry_run"}
        save_json(path, data)
        self.saved.config(text=f"Saved to {path}", fg=GREEN)

    def refresh(self, now):
        # Reflect changes made elsewhere (keys, voice, guided setup).
        for name, var in self.vars.items():
            value = getattr(self.app.cfg, name)
            if var.get() != value:
                var.set(value)

    def close_window(self):
        self.close()
        self.ui.settings = None

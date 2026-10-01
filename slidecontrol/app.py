"""The camera loop that ties all features together."""

import dataclasses
import logging
import time
from collections import Counter

import numpy as np

from .config import Config, list_profiles, load_json, load_profile, merge, save_json, user_config_path
from .feedback import Feedback
from .geometry import (LEFT_EYE, MOUTH, NOSE_TIP, RIGHT_EYE, extract_features, face_size)
from .gestures import Calibrator, Dispatcher, GestureEngine
from .hands import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, HandController
from .inputs import InputSender
from .keyboard import ScanKeyboard
from .motion import GazeCalibration, GazeModel, GazePointer, HeadScroller
from .setup_wizard import GuidedSetup
from .stats import build_session, save_session
from .system import Tray
from .ui import TkUI
from .voice import VoiceListener
from .wellness import Presence, Wellness

log = logging.getLogger("slidecontrol")

# BGR colours for the OpenCV preview.
GREEN = (80, 220, 80)
RED = (60, 60, 230)
YELLOW = (0, 210, 240)
WHITE = (240, 240, 240)
GREY = (140, 140, 140)
CYAN = (230, 200, 60)

PREVIEW_KEYS = ("q quit | p pause | c calibrate | g setup | s settings | k keyboard | l laser | "
                "o overlay | h hand | t head-scroll | e gaze | v voice | 1-9 profile")


class App:
    def __init__(self, cfg, base_data=None, cli_data=None, config_path=None):
        self.cfg = cfg
        self.base_data = base_data or {}
        self.cli_data = cli_data or {}
        self.config_path = config_path or user_config_path()
        self.feedback = Feedback(cfg)
        self.sender = InputSender(cfg.input_backend, cfg.dry_run)
        self.dispatcher = Dispatcher(cfg, self.sender, self.feedback)
        self.engine = GestureEngine(cfg)
        self.hand = HandController(cfg, self.sender)
        self.head_scroller = HeadScroller(cfg, self.sender)
        self.presence = Presence(cfg)
        self.wellness = Wellness(cfg, self.feedback)
        self.calibrator = Calibrator(cfg.calibration_seconds)
        self.gaze_model = GazeModel.load()
        self.gaze_pointer = GazePointer(cfg, self.gaze_model)
        self.gaze_calibration = None
        self.setup = None
        self.keyboard = None
        self.voice = None
        self.ui = None
        self.tray = None
        self.smoothed = None
        self.face_count = 0
        self.fps = 0.0
        self.start_time = time.monotonic()
        self.start_wall = time.time()
        self.near_misses = Counter()
        self.message = ("", 0.0)
        self.quit_requested = False
        self.last_voice = ("", 0.0)

    # ------------------------------------------------------------------ state

    @property
    def baseline(self):
        return self.engine.baseline

    def say(self, text, now=None):
        """Short message shown in the preview and logged."""
        log.info(text)
        self.message = (text, now if now is not None else time.monotonic())

    def status_text(self):
        if self.setup is not None:
            return "GUIDED SETUP", "#ffca28"
        if self.gaze_calibration is not None:
            return "GAZE CALIBRATION", "#ffca28"
        if self.baseline is None:
            return "CALIBRATING", "#ffca28"
        if self.dispatcher.auto_paused:
            return f"PAUSED ({self.dispatcher.auto_paused})", "#ef5350"
        if self.dispatcher.paused:
            return "PAUSED", "#ef5350"
        return "ACTIVE", "#66bb6a"

    def apply_config(self):
        """Re-applies settings after they changed (settings window, profile, setup)."""
        self.near_misses += self.engine.near_misses()
        self.engine = GestureEngine(self.cfg, self.engine.baseline)
        if self.hand.enabled != self.cfg.hand_control:
            self.hand.enabled = self.cfg.hand_control
            self.hand.reset()
        if self.ui:
            self.ui.set_overlay(self.cfg.overlay)
        if self.cfg.voice and self.voice is None:
            self.start_voice()
        elif not self.cfg.voice and self.voice is not None:
            self.voice.stop()
            self.voice = None
        if self.cfg.gaze and self.gaze_model is None and self.gaze_calibration is None:
            self.start_gaze_calibration()
        self.hand.pointer_enabled = not (self.cfg.gaze and self.gaze_model is not None)

    def switch_profile(self, name):
        try:
            data = merge(merge(self.base_data, load_profile(name)), self.cli_data)
            data["profile"] = name
            new = Config.from_dict(data)
        except (OSError, ValueError) as exc:
            self.say(f"Profile error: {exc}")
            return
        for f in dataclasses.fields(Config):
            setattr(self.cfg, f.name, getattr(new, f.name))
        self.apply_config()
        self.feedback.play("toggle")
        self.say(f"Profile: {name}")

    # ------------------------------------------------------------ modes

    def recalibrate(self):
        self.say("Calibrating: look at the screen with eyes open and a neutral face")
        self.engine.baseline = None
        self.calibrator = Calibrator(self.cfg.calibration_seconds)

    def start_setup(self):
        self.say("Guided setup started")
        self.setup = GuidedSetup()
        if self.ui:
            self.ui.show_setup(True)

    def finish_setup(self):
        baseline, settings = self.setup.result
        self.setup = None
        if self.ui:
            self.ui.show_setup(False)
        for k, v in settings.items():
            setattr(self.cfg, k, v)
        self.engine.baseline = baseline
        self.apply_config()
        try:
            stored = load_json(self.config_path) if self.config_path.is_file() else {}
            save_json(self.config_path, {**stored, **settings})
            where = f" and saved to {self.config_path}"
        except OSError as exc:
            where = f" (could not save: {exc})"
        self.feedback.play("calibrated")
        self.feedback.notify("Setup complete", ", ".join(f"{k}={v}" for k, v in settings.items()))
        self.say(f"Setup complete{where}")

    def start_gaze_calibration(self):
        if self.ui is None:
            self.say("Gaze calibration needs the on-screen windows (Tk)")
            self.cfg.gaze = False
            return
        self.say("Gaze calibration: follow the dots")
        self.gaze_calibration = GazeCalibration()
        self.ui.show_gaze_calibration(True)

    def cancel_gaze_calibration(self):
        self.gaze_calibration = None
        if self.ui:
            self.ui.show_gaze_calibration(False)
        if self.gaze_model is None:
            self.cfg.gaze = False
        self.say("Gaze calibration cancelled")

    def finish_gaze_calibration(self):
        model = self.gaze_calibration.fit()
        self.gaze_calibration = None
        if self.ui:
            self.ui.show_gaze_calibration(False)
        if model is None:
            self.cfg.gaze = False
            self.say("Gaze calibration failed: face not visible for enough dots")
            return
        model.save()
        self.gaze_model = model
        self.gaze_pointer = GazePointer(self.cfg, model)
        self.cfg.gaze = True
        self.apply_config()
        self.feedback.play("calibrated")
        self.say("Gaze calibrated: your eyes now move the pointer")

    def toggle_keyboard(self):
        if self.keyboard is None:
            self.keyboard = ScanKeyboard()
            self.say("Keyboard: wink left = select, wink right = back")
        else:
            self.keyboard = None
        if self.ui:
            self.ui.set_keyboard(self.keyboard is not None)

    def keyboard_press(self, label):
        if self.keyboard:
            self.keyboard_action(self.keyboard.press(label))

    def keyboard_action(self, action):
        if action == "@close":
            self.toggle_keyboard()
        elif action:
            self.sender.send(action)

    def toggle_laser(self):
        self.hand.laser = not self.hand.laser
        if self.ui:
            self.ui.set_laser(self.hand.laser)
        self.say(f"Laser pointer {'on' if self.hand.laser else 'off'} (point with your index finger)")

    def start_voice(self):
        self.voice = VoiceListener(self.cfg.voice_model_path, self.cfg.voice_commands, list_profiles())
        self.voice.start()

    def handle_internal(self, action):
        if not action:
            return
        if action in ("@pause", "@pause_on", "@resume"):
            return
        handlers = {
            "@recalibrate": self.recalibrate,
            "@laser": self.toggle_laser,
            "@keyboard": self.toggle_keyboard,
            "@overlay": lambda: self.toggle_setting("overlay"),
            "@settings": lambda: self.ui and self.ui.open_settings(),
            "@setup": self.start_setup,
            "@gaze_calibrate": self.start_gaze_calibration,
        }
        if action in handlers:
            handlers[action]()
        elif action.startswith("@profile:"):
            self.switch_profile(action.split(":", 1)[1])
        else:
            log.warning("Unknown action %s", action)

    def toggle_setting(self, name):
        setattr(self.cfg, name, not getattr(self.cfg, name))
        self.apply_config()
        self.say(f"{name.replace('_', ' ')}: {'on' if getattr(self.cfg, name) else 'off'}")

    # ------------------------------------------------------------- per frame

    def step(self, faces, hands, frame_w, frame_h, now):
        """Processes one frame. `faces`/`hands` are lists of landmark arrays in pixels."""
        self.face_count = len(faces)
        features = extract_features(faces[0]) if faces else None
        self.smoothed = None if features is None else features.blend(self.smoothed, self.cfg.smoothing)

        reason = self.presence.update(len(faces), now)
        if reason != self.dispatcher.auto_paused:
            self.dispatcher.auto_paused = reason
            self.feedback.play("pause" if reason else "resume")
            self.say(f"Auto-paused: {reason}" if reason else "Resumed", now)

        for event in self.hand.update(hands, frame_w, frame_h, now, active=self.dispatcher.active):
            self.handle_internal(self.dispatcher.handle(event, now))
        self.process_voice(now)
        if self.keyboard:
            self.keyboard.tick(now)

        if self.setup is not None:
            self.setup.update(self.smoothed, now)
            if self.setup.finished:
                self.finish_setup()
            return
        if self.gaze_calibration is not None:
            self.gaze_calibration.update(self.smoothed, now)
            if self.gaze_calibration.finished(now):
                self.finish_gaze_calibration()
            return
        if self.baseline is None:
            if features is not None:
                self.calibrator.add(features, now)
                baseline = self.calibrator.result(now)
                if baseline is not None:
                    self.engine.baseline = baseline
                    self.feedback.play("calibrated")
                    self.say("Calibrated", now)
            return

        both_closed = False
        if self.smoothed is not None:
            left, right = self.engine.eye_ratios(self.smoothed)
            both_closed = left < self.cfg.close_ratio and right < self.cfg.close_ratio
        self.wellness.update(features is not None, both_closed, now)

        for gesture in self.engine.update(self.smoothed, now):
            if self.keyboard and gesture in ("wink_left", "wink_right") and self.dispatcher.active:
                if gesture == "wink_left":
                    self.keyboard_action(self.keyboard.select(now))
                else:
                    self.keyboard.back(now)
                continue
            self.handle_internal(self.dispatcher.handle(gesture, now))

        active = self.dispatcher.active
        if self.cfg.head_scroll:
            self.head_scroller.update(self.smoothed, self.baseline, now, active=active and not self.keyboard)
        if self.cfg.gaze and self.gaze_model is not None and active:
            point, click = self.gaze_pointer.update(self.smoothed, now)
            if point is not None:
                self.sender.move_to(*point)
            if click:
                self.sender.click("left")

    def process_voice(self, now):
        if self.voice is None:
            return
        while not self.voice.results.empty():
            text, action = self.voice.results.get()
            self.last_voice = (text, now)
            if action:
                self.handle_internal(self.dispatcher.perform(f"voice: {text}", action, now, cooldown=False))
            else:
                log.debug("Voice: no command for %r", text)

    def process_tray(self):
        if self.tray is None:
            return
        self.tray.set_status(self.status_text()[0].capitalize())
        while not self.tray.commands.empty():
            cmd = self.tray.commands.get()
            if cmd == "quit":
                self.quit_requested = True
            elif cmd == "toggle":
                self.dispatcher.set_paused(not self.dispatcher.paused)
            elif cmd == "recalibrate":
                self.recalibrate()
            elif cmd == "setup":
                self.start_setup()
            elif cmd == "gaze":
                self.start_gaze_calibration()
            elif cmd == "settings" and self.ui:
                self.ui.open_settings()
            elif cmd == "keyboard":
                self.toggle_keyboard()
            elif cmd == "overlay":
                self.toggle_setting("overlay")
            elif cmd == "laser":
                self.toggle_laser()

    def handle_key(self, key):
        """Keyboard shortcuts in the preview window. Returns False to quit."""
        if key in (ord("q"), 27):
            return False
        actions = {
            ord("p"): lambda: self.dispatcher.set_paused(not self.dispatcher.paused),
            ord("c"): self.recalibrate,
            ord("g"): self.start_setup,
            ord("s"): lambda: self.ui and self.ui.open_settings(),
            ord("k"): self.toggle_keyboard,
            ord("l"): self.toggle_laser,
            ord("o"): lambda: self.toggle_setting("overlay"),
            ord("h"): lambda: self.toggle_setting("hand_control"),
            ord("t"): lambda: self.toggle_setting("head_scroll"),
            ord("v"): lambda: self.toggle_setting("voice"),
            ord("m"): lambda: self.toggle_setting("show_mesh"),
            ord("e"): self.start_gaze_calibration,
            ord("d"): self.toggle_dry_run,
        }
        if key in actions:
            actions[key]()
        elif ord("1") <= key <= ord("9"):
            profiles = list_profiles()
            index = key - ord("1")
            if index < len(profiles):
                self.switch_profile(profiles[index])
        return True

    def toggle_dry_run(self):
        self.sender.dry_run = not self.sender.dry_run
        self.say(f"Dry-run {'on' if self.sender.dry_run else 'off'}")

    # ------------------------------------------------------------- main loop

    def run(self, start_setup=False, start_gaze=False, tray=False):
        import cv2
        import mediapipe as mp

        cap = cv2.VideoCapture(self.cfg.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        if not cap.isOpened():
            log.error("Could not open camera %s (is another program using it?)", self.cfg.camera)
            return 1

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        hands_model = mp.solutions.hands.Hands(
            max_num_hands=2, model_complexity=0,
            min_detection_confidence=0.6, min_tracking_confidence=0.5)
        self.sender.prepare()
        self.ui = TkUI.create(self)
        if tray:
            self.tray = Tray()
            if not self.tray.start():
                self.tray = None
        self.apply_config()
        if start_setup:
            self.start_setup()
        else:
            self.recalibrate()
        if start_gaze:
            self.start_gaze_calibration()

        window = "Face Slide Control"
        prev = time.monotonic()
        try:
            while not self.quit_requested:
                ok, frame = cap.read()
                if not ok:
                    log.error("Camera stopped delivering frames")
                    break
                now = time.monotonic()
                self.fps = 0.9 * self.fps + 0.1 / max(now - prev, 1e-6)
                prev = now

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                faces = []
                result = face_mesh.process(rgb)
                for face in result.multi_face_landmarks or []:
                    faces.append(np.array([(p.x * w, p.y * h) for p in face.landmark]))
                faces.sort(key=face_size, reverse=True)  # the closest face is the user
                hands = []
                if self.hand.enabled:
                    hand_result = hands_model.process(rgb)
                    for hand in hand_result.multi_hand_landmarks or []:
                        hands.append(np.array([(p.x * w, p.y * h) for p in hand.landmark]))

                self.step(faces, hands, w, h, now)

                if self.cfg.show_preview:
                    # Landmarks are computed on the unmirrored frame so that eye
                    # sides stay anatomical; mirroring is for display only.
                    if faces and self.cfg.show_mesh:
                        self.draw_face(cv2, frame, faces[0])
                    for lm in hands:
                        self.draw_hand(cv2, frame, lm)
                    if self.cfg.mirror:
                        frame = cv2.flip(frame, 1)
                    self.draw_hud(cv2, frame, now, bool(faces))
                    cv2.imshow(window, frame)
                    if not self.handle_key(cv2.waitKey(1) & 0xFF):
                        break
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                if self.ui:
                    self.ui.update(now)
                self.process_tray()
        except KeyboardInterrupt:
            pass
        finally:
            self.hand.reset()  # never leave a mouse button held down
            if self.voice:
                self.voice.stop()
            if self.tray:
                self.tray.stop()
            if self.ui:
                self.ui.destroy()
            cap.release()
            face_mesh.close()
            hands_model.close()
            cv2.destroyAllWindows()
            self.save_stats()
            self.sender.close()
        return 0

    def save_stats(self):
        self.near_misses += self.engine.near_misses()
        counts = self.dispatcher.counts
        session = build_session(self.start_wall, time.monotonic() - self.start_time, counts,
                                self.hand.clicks, self.near_misses, self.wellness.blinks.total,
                                self.cfg.profile or "default")
        try:
            save_session(session)
        except OSError as exc:
            log.warning("Could not save stats: %s", exc)
        summary = counts + self.hand.clicks
        if summary:
            log.info("Session summary: %s", ", ".join(f"{g}={n}" for g, n in summary.most_common()))

    # ----------------------------------------------------------------- drawing

    def draw_face(self, cv2, frame, lm):
        for idx in RIGHT_EYE + LEFT_EYE + MOUTH + [NOSE_TIP]:
            x, y = lm[idx]
            cv2.circle(frame, (int(x), int(y)), 2, YELLOW, -1)
        for eye in (RIGHT_EYE, LEFT_EYE):
            cv2.polylines(frame, [lm[eye].astype(np.int32)], True, GREEN, 1)

    def draw_hand(self, cv2, frame, lm):
        for x, y in lm:
            cv2.circle(frame, (int(x), int(y)), 2, WHITE, -1)
        color = RED if self.hand.left_down else GREEN
        cv2.line(frame, tuple(lm[THUMB_TIP].astype(int)), tuple(lm[INDEX_TIP].astype(int)), color, 2)
        cv2.circle(frame, tuple(lm[MIDDLE_MCP].astype(int)), 7, YELLOW, 2)

    def draw_hud(self, cv2, frame, now, face_found):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        def text(s, org, color=WHITE, scale=0.5, thick=1):
            cv2.putText(frame, s, org, font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(frame, s, org, font, scale, color, thick, cv2.LINE_AA)

        def bar(x, y, value, label, marks=(), color=GREEN, width=140):
            cv2.rectangle(frame, (x, y), (x + width, y + 10), GREY, 1)
            cv2.rectangle(frame, (x, y), (x + int(width * min(max(value, 0), 1)), y + 10), color, -1)
            for m in marks:
                mx = x + int(width * m)
                cv2.line(frame, (mx, y - 2), (mx, y + 12), RED, 1)
            text(label, (x + width + 8, y + 10))

        status, _ = self.status_text()
        color = GREEN if status == "ACTIVE" else (YELLOW if status in ("CALIBRATING", "GUIDED SETUP",
                                                                         "GAZE CALIBRATION") else RED)
        if self.sender.dry_run:
            status += "  [dry-run]"
        text(status, (10, 25), color, 0.6, 2)
        text(f"{self.fps:4.1f} fps", (w - 90, 25), GREY)
        text(f"profile: {self.cfg.profile or 'default'}", (w - 200, 45), GREY, 0.45)

        modes = [name for name, on in [
            ("hand", self.hand.enabled), ("head-scroll", self.cfg.head_scroll),
            ("gaze", self.cfg.gaze and self.gaze_model is not None), ("laser", self.hand.laser),
            ("keyboard", self.keyboard is not None),
            ("voice" + ("" if self.voice is None or self.voice.listening else " (starting)"), self.voice is not None),
        ] if on]
        text("on: " + (", ".join(modes) or "-"), (w - 200, 63), CYAN, 0.42)
        if self.hand.enabled and self.hand.pose:
            label = "LEFT BUTTON DOWN" if self.hand.left_down else self.hand.pose
            text(f"hand: {label}", (w - 200, 81), RED if self.hand.left_down else GREEN, 0.45)

        if self.setup is not None and not self.setup.finished:
            _, instruction, _ = self.setup.current()
            text(instruction, (10, 55), YELLOW, 0.55, 2)
            bar(10, 65, self.setup.progress(now), f"step {self.setup.index + 1}/{len(self.setup.steps)}",
                color=YELLOW)
        elif not face_found:
            text("No face detected", (10, 55), RED, 0.6, 2)
        elif self.baseline is None:
            bar(10, 45, self.calibrator.progress(now), "calibration", color=YELLOW)
        elif self.smoothed is not None:
            left, right = self.engine.eye_ratios(self.smoothed)
            marks = (self.cfg.close_ratio / 1.5, self.cfg.open_ratio / 1.5)
            bar(10, 45, left / 1.5, f"left eye  {left:.2f}", marks)
            bar(10, 65, right / 1.5, f"right eye {right:.2f}", marks)
            y = 95
            for gesture, p in self.engine.progress(now).items():
                if p > 0:
                    bar(10, y, p, gesture, color=YELLOW, width=80)
                    y += 20
            if self.cfg.head_scroll and self.head_scroller.speed:
                text(f"head scroll {self.head_scroller.speed:+.0f}/s", (10, y + 10), CYAN)

        if self.voice is not None and now - self.last_voice[1] < 3:
            text(f'heard: "{self.last_voice[0]}"', (10, h - 88), CYAN, 0.5)
        if self.message[0] and now - self.message[1] < 3:
            text(self.message[0], (10, h - 64), CYAN, 0.5)
        d = self.dispatcher
        if d.last_action and now - d.last_action_time < 2.0:
            text(d.last_action, (10, h - 40), YELLOW, 0.7, 2)
        text(PREVIEW_KEYS, (10, h - 12), GREY, 0.33)


"""Face gestures, dispatching, input and configuration."""

import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slidecontrol.app import App  # noqa: E402
from slidecontrol.cli import autostart_args, build_config, parse_args  # noqa: E402
from slidecontrol.config import FACE_GESTURES, Config, list_profiles  # noqa: E402
from slidecontrol.geometry import (LEFT_EYE_OUTER, RIGHT_EYE_OUTER, FaceFeatures,  # noqa: E402
                                   extract_features, eye_aspect_ratio)
from slidecontrol.gestures import Dispatcher, GestureEngine, HoldTrigger  # noqa: E402
from slidecontrol.inputs import InputSender  # noqa: E402

DEFAULTS = dict(ear_left=0.3, ear_right=0.3, mar=0.05, yaw=0.5, roll=0.0, pitch=0.5, brow=0.5,
                smile=1.0, gaze_x=0.5, gaze_y=0.0)
BASE = FaceFeatures(**DEFAULTS)


def face(**kw):
    return FaceFeatures(**{**DEFAULTS, **kw})


class FakeSender:
    def __init__(self):
        self.sent = []
        self.dry_run = False

    def send(self, action):
        self.sent.append(action)

    def move(self, dx, dy):
        self.sent.append(("move", dx, dy))

    def move_to(self, x, y):
        self.sent.append(("move_to", round(x, 3), round(y, 3)))

    def scroll(self, notches):
        self.sent.append(("scroll", notches))

    def button(self, name, pressed):
        self.sent.append((name, "down" if pressed else "up"))

    def click(self, name):
        self.button(name, True)
        self.button(name, False)


class SilentFeedback:
    def __init__(self):
        self.played, self.notified = [], []

    def play(self, event):
        self.played.append(event)

    def notify(self, title, body="", urgent=False):
        self.notified.append(title)


def run(engine, features, start, duration, step=1 / 30):
    """Feeds the same features for `duration` seconds; returns fired gestures."""
    fired, t = [], start
    while t <= start + duration + 1e-9:
        fired += engine.update(features, t)
        t += step
    return fired


def isolated_home(test):
    """Points XDG config/data dirs at a temp dir for the duration of a test."""
    tmp = tempfile.TemporaryDirectory()
    test.addCleanup(tmp.cleanup)
    patcher = unittest.mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": tmp.name + "/config",
                                                    "XDG_DATA_HOME": tmp.name + "/data"})
    patcher.start()
    test.addCleanup(patcher.stop)
    return Path(tmp.name)


class GeometryTest(unittest.TestCase):
    def test_ear_is_scale_invariant(self):
        eye = np.array([[0, 0], [2, -1], [4, -1], [6, 0], [4, 1], [2, 1]], float)
        self.assertAlmostEqual(eye_aspect_ratio(eye), 2 / 6)
        self.assertAlmostEqual(eye_aspect_ratio(eye * 5), eye_aspect_ratio(eye))

    def test_roll_sign(self):
        lm = np.zeros((478, 2))
        lm[RIGHT_EYE_OUTER] = (100, 100)
        lm[LEFT_EYE_OUTER] = (200, 130)  # left eye lower in the image -> tilt to the left
        self.assertGreater(extract_features(lm).roll, 0)


class HoldTriggerTest(unittest.TestCase):
    def test_fires_once_after_hold_and_rearms(self):
        t = HoldTrigger(0.2)
        self.assertFalse(t.update(True, 0.0))
        self.assertFalse(t.update(True, 0.1))
        self.assertTrue(t.update(True, 0.2))
        self.assertFalse(t.update(True, 1.0))
        t.update(False, 1.1)
        self.assertFalse(t.update(True, 1.2))
        self.assertTrue(t.update(True, 1.5))

    def test_counts_near_misses(self):
        t = HoldTrigger(1.0)
        t.update(True, 0.0)
        t.update(True, 0.7)
        t.update(False, 0.8)  # held 70 % of the time, released
        t.update(True, 1.0)
        t.update(False, 1.1)  # held 0 %, not counted
        self.assertEqual(t.near_misses, 1)


class GestureEngineTest(unittest.TestCase):
    def make(self, gestures=FACE_GESTURES, **cfg):
        engine = GestureEngine(Config(gestures=list(gestures), **cfg))
        engine.baseline = BASE
        return engine

    def test_wink_left(self):
        self.assertEqual(run(self.make(), face(ear_left=0.1), 0, 0.4), ["wink_left"])

    def test_wink_right(self):
        self.assertEqual(run(self.make(), face(ear_right=0.1), 0, 0.4), ["wink_right"])

    def test_wink_with_other_eye_squinting(self):
        self.assertEqual(run(self.make(), face(ear_left=0.1, ear_right=0.21), 0, 0.4), ["wink_left"])

    def test_held_scroll_wink_repeats(self):
        # first step at 0.2 s, repeats start 0.5 s later, then every 0.12 s
        self.assertEqual(run(self.make(), face(ear_left=0.1), 0, 1.0), ["wink_left"] * 4)

    def test_no_repeat_when_disabled(self):
        self.assertEqual(run(self.make(repeat_enabled=False), face(ear_left=0.1), 0, 1.0), ["wink_left"])

    def test_short_wink_is_ignored(self):
        self.assertEqual(run(self.make(), face(ear_left=0.1), 0, 0.1), [])

    def test_natural_blink_does_not_change_slides(self):
        self.assertEqual(run(self.make(), face(ear_left=0.1, ear_right=0.1), 0, 0.4), [])

    def test_long_blink(self):
        self.assertEqual(run(self.make(), face(ear_left=0.1, ear_right=0.1), 0, 1.5), ["long_blink"])

    def test_head_mouth_brow_smile(self):
        cases = [(face(yaw=0.7), "head_left"), (face(yaw=0.3), "head_right"),
                 (face(roll=25), "tilt_left"), (face(roll=-25), "tilt_right"),
                 (face(mar=0.8), "mouth_open"), (face(brow=0.65), "brow_raise"), (face(smile=1.4), "smile")]
        for features, expected in cases:
            with self.subTest(expected):
                self.assertEqual(run(self.make(), features, 0, 0.7)[:1], [expected])

    def test_open_mouth_is_not_a_smile(self):
        self.assertNotIn("smile", run(self.make(), face(smile=1.4, mar=0.8), 0, 1))

    def test_disabled_gestures_never_fire(self):
        self.assertEqual(run(self.make(["wink_left"]), face(yaw=0.8), 0, 1), [])

    def test_face_lost_resets_progress(self):
        engine = self.make()
        engine.update(face(ear_left=0.1), 0.0)
        engine.update(None, 0.15)
        self.assertEqual(engine.update(face(ear_left=0.1), 0.25), [])


class DispatcherTest(unittest.TestCase):
    def setUp(self):
        self.sender = FakeSender()
        self.feedback = SilentFeedback()
        self.d = Dispatcher(Config(), self.sender, self.feedback)

    def test_sends_mapped_key_and_respects_cooldown(self):
        self.d.handle("wink_left", 0.0)
        self.d.handle("wink_right", 0.3)
        self.d.handle("wink_right", 1.0)
        self.assertEqual(self.sender.sent, ["down", "up"])
        self.assertEqual(self.d.position, 0)

    def test_repeats_skip_cooldown_and_wheel_scroll(self):
        self.d.handle("wink_left", 0.0)
        self.d.handle("wink_left", 0.12)
        self.d.handle("wink_right", 0.3)  # a different gesture still waits for the cooldown
        self.d.cfg.actions["wink_right"] = "@scroll_up"
        self.d.handle("wink_right", 1.0)
        self.assertEqual(self.sender.sent, ["down", "down", ("scroll", 3)])

    def test_pause_blocks_everything_but_pause_actions(self):
        self.assertEqual(self.d.handle("long_blink", 0.0), "@pause")
        self.assertTrue(self.d.paused)
        self.d.handle("wink_left", 1.0)
        self.assertEqual(self.sender.sent, [])
        self.d.handle("fist_hold", 2.0)  # @resume
        self.assertFalse(self.d.paused)
        self.d.handle("wink_left", 3.0)
        self.assertEqual(self.sender.sent, ["down"])
        self.assertEqual(self.feedback.played[:2], ["pause", "resume"])

    def test_auto_pause(self):
        self.d.auto_paused = "away"
        self.d.handle("wink_left", 0.0)
        self.assertEqual(self.sender.sent, [])

    def test_goto_page(self):
        self.d.cfg.goto_prefix = "ctrl+l"
        self.d.perform("voice", "@goto:12", 0.0, cooldown=False)
        self.assertEqual(self.sender.sent, ["ctrl+l", "1", "2", "enter"])

    def test_ui_actions_are_returned_to_the_app(self):
        self.assertEqual(self.d.perform("voice", "@keyboard", 0.0), "@keyboard")


class InputSenderTest(unittest.TestCase):
    def test_backend_errors_do_not_raise(self):
        sender = InputSender()

        class Broken:
            def send(self, action):
                raise RuntimeError("display unavailable")

        sender._backend = Broken()
        with self.assertLogs("slidecontrol", "ERROR"):
            sender.send("down")

    def test_missing_backend_disables_input_quietly(self):
        sender = InputSender(backend="pyautogui")
        with unittest.mock.patch("slidecontrol.inputs.create_backend", side_effect=OSError("no display")) as cb:
            with self.assertLogs("slidecontrol", "ERROR"):
                sender.send("down")
            sender.send("down")
            sender.move(1, 1)
        self.assertEqual(cb.call_count, 1)


class AppFlowTest(unittest.TestCase):
    def setUp(self):
        isolated_home(self)

    def make_app(self, **cfg):
        app = App(Config(calibration_seconds=1.0, smoothing=0.0, sounds=False, notifications=False,
                         hand_control=False, **cfg))
        app.sender = app.dispatcher.sender = app.hand.sender = app.head_scroller.sender = FakeSender()
        return app

    def feed(self, app, lm_fn, seconds, t):
        while seconds > 0:
            app.step(lm_fn(), [], 640, 480, t)
            t += 1 / 30
            seconds -= 1 / 30
        return t

    def test_calibrates_then_detects(self):
        app = self.make_app()
        neutral = [np.zeros((478, 2))]
        t = 0.0
        with unittest.mock.patch("slidecontrol.app.extract_features") as ef:
            ef.return_value = face(ear_left=0.32, ear_right=0.28)
            t = self.feed(app, lambda: neutral, 1.2, t)
            self.assertAlmostEqual(app.baseline.ear_left, 0.32)
            ef.return_value = face(ear_left=0.1, ear_right=0.28)
            self.feed(app, lambda: neutral, 0.4, t)
        self.assertEqual(app.sender.sent, ["down"])

    def test_auto_pauses_when_face_leaves(self):
        app = self.make_app(away_seconds=1.0)
        t = self.feed(app, lambda: [], 1.2, 0.0)
        self.assertEqual(app.dispatcher.auto_paused, "away")
        with unittest.mock.patch("slidecontrol.app.extract_features", return_value=BASE):
            self.feed(app, lambda: [np.zeros((478, 2))], 0.1, t)
        self.assertEqual(app.dispatcher.auto_paused, "")

    def test_profile_switch_keeps_object_identity(self):
        app = self.make_app()
        cfg = app.cfg
        app.switch_profile("slides")
        self.assertIs(app.cfg, cfg)
        self.assertEqual(app.cfg.actions["wink_left"], "pagedown")
        self.assertEqual(app.cfg.profile, "slides")


class SettingsSaveTest(unittest.TestCase):
    def test_save_skips_command_line_options(self):
        home = isolated_home(self)
        from slidecontrol.ui import SettingsWindow
        app = App(Config(dry_run=True, show_preview=False, cooldown=0.9),
                  cli_data={"dry_run": True, "show_preview": False}, config_path=home / "cfg.json")
        window = SettingsWindow.__new__(SettingsWindow)  # no Tk needed for saving
        window.app = app
        window.saved = unittest.mock.Mock()
        window.save()
        saved = json.loads((home / "cfg.json").read_text())
        self.assertEqual(saved["cooldown"], 0.9)
        self.assertNotIn("dry_run", saved)
        self.assertNotIn("show_preview", saved)


class ConfigTest(unittest.TestCase):
    def setUp(self):
        isolated_home(self)

    def test_layers_file_profile_and_cli(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"cooldown": 1.0, "actions": {"wink_left": "right"}}, fh)
        self.addCleanup(os.unlink, fh.name)
        cfg, *_ = build_config(parse_args(["--config", fh.name, "--gestures", "all"]))
        self.assertEqual(cfg.cooldown, 1.0)
        self.assertEqual(cfg.actions["wink_left"], "right")
        self.assertEqual(cfg.actions["wink_right"], "up")
        self.assertEqual(cfg.gestures, FACE_GESTURES)

        cfg, *_ = build_config(parse_args(["--config", fh.name, "--profile", "pdf", "--no-hand"]))
        self.assertEqual(cfg.actions["wink_left"], "down")  # profile beats file
        self.assertTrue(cfg.head_scroll)
        self.assertFalse(cfg.hand_control)  # CLI beats everything

    def test_shipped_profiles_are_valid(self):
        self.assertEqual(set(list_profiles()) >= {"pdf", "slides", "video"}, True)
        for name in list_profiles():
            with self.subTest(name):
                build_config(parse_args(["--profile", name]))

    def test_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"gestures": ["sneeze"]})
        with self.assertRaises(ValueError):
            Config.from_dict({"colour": "red"})

    def test_autostart_args(self):
        self.assertEqual(autostart_args(["--install-autostart", "--profile", "pdf"]),
                         ["--profile", "pdf", "--tray", "--no-preview"])


if __name__ == "__main__":
    unittest.main()

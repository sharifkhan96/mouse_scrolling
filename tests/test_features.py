"""Hands, head scroll, gaze, voice, setup wizard, keyboard, wellness, stats, autostart."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_core import BASE, FakeSender, SilentFeedback, face, isolated_home  # noqa: E402

from slidecontrol.config import Config, default_voice_commands  # noqa: E402
from slidecontrol.hands import (INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_PIP,  # noqa: E402
                                MIDDLE_TIP, PINKY_PIP, PINKY_TIP, RING_PIP, RING_TIP, THUMB_TIP, WRIST,
                                HandController, hand_pose)
from slidecontrol.keyboard import ScanKeyboard  # noqa: E402
from slidecontrol.motion import (GazeCalibration, GazeModel, GazePointer, HeadScroller,  # noqa: E402
                                 gaze_model_path)
from slidecontrol.setup_wizard import GuidedSetup, derive_settings  # noqa: E402
from slidecontrol.stats import build_session, load_sessions, save_session, summarize  # noqa: E402
from slidecontrol.system import autostart_path, install_autostart, remove_autostart  # noqa: E402
from slidecontrol.voice import grammar, parse_command, words_to_number  # noqa: E402
from slidecontrol.wellness import BlinkMonitor, Presence, Wellness  # noqa: E402


def hand(x=300.0, y=200.0, pose="point", pinch=None):
    """Synthetic hand, ~100 px from wrist to middle knuckle, knuckle at (x, y)."""
    lm = np.zeros((21, 2))
    lm[:] = (x, y)
    lm[WRIST] = (x, y + 100)
    lm[INDEX_MCP] = (x - 20, y)
    lm[MIDDLE_MCP] = (x, y)
    extended = {"point": [1, 0, 0, 0], "palm": [1, 1, 1, 1], "fist": [0, 0, 0, 0], "two": [1, 1, 0, 0]}[pose]
    for (tip, pip), dx, ext in zip([(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP), (RING_TIP, RING_PIP),
                                    (PINKY_TIP, PINKY_PIP)], (-20, 0, 20, 40), extended):
        lm[pip] = (x + dx, y - 40)
        lm[tip] = (x + dx, y - 90) if ext else (x + dx, y + 10)
    lm[THUMB_TIP] = (x - 70, y + 20)
    if pinch == "index":
        lm[INDEX_TIP] = (x - 50, y - 60)
        lm[THUMB_TIP] = lm[INDEX_TIP] + (5, 0)
    elif pinch == "middle":
        lm[THUMB_TIP] = lm[MIDDLE_TIP] + (5, 0)
    return lm


class HandControllerTest(unittest.TestCase):
    def setUp(self):
        self.sender = FakeSender()
        self.hc = HandController(Config(pointer_smoothing=0.0), self.sender)
        self.t = 0.0

    def feed(self, *hands, active=True, dt=1 / 30):
        events = self.hc.update(list(hands), 640, 480, self.t, active)
        self.t += dt
        return events

    def moves(self):
        return [e for e in self.sender.sent if e[0] == "move"]

    def test_poses(self):
        for pose in ("point", "palm", "fist", "two"):
            self.assertEqual(hand_pose(hand(pose=pose)), pose)

    def test_moving_hand_moves_pointer_mirrored(self):
        self.feed(hand(300))
        self.feed(hand(332))  # hand moves right in the image = user's left
        self.assertEqual(self.moves(), [("move", -100, 0)])

    def test_jitter_below_deadzone_is_ignored(self):
        self.feed(hand(300))
        self.feed(hand(300.3))
        self.assertEqual(self.moves(), [])

    def test_pinch_presses_and_releases_left_button(self):
        self.feed(hand())
        self.feed(hand(pinch="index"))
        self.feed(hand(pinch="index"))
        self.feed(hand())
        self.assertEqual(self.sender.sent, [("left", "down"), ("left", "up")])

    def test_middle_pinch_right_clicks_once(self):
        for _ in range(3):
            self.feed(hand(pinch="middle"))
        self.assertEqual(self.sender.sent, [("right", "down"), ("right", "up")])

    def test_fist_is_a_clutch(self):
        self.feed(hand(300, pose="fist"))
        self.feed(hand(400, pose="fist"))
        self.assertEqual(self.moves(), [])

    def test_losing_hand_or_pausing_releases_button(self):
        self.feed(hand(pinch="index"))
        self.feed()
        self.feed(hand(pinch="index"))
        self.feed(hand(pinch="index"), active=False)
        self.assertEqual(self.sender.sent, [("left", "down"), ("left", "up")] * 2)

    def test_palm_swipe(self):
        events = []
        for x in range(200, 420, 40):  # fast movement to the right of the image = user's left
            events += self.feed(hand(x, pose="palm"))
        self.assertEqual(events, ["swipe_left"])
        self.assertEqual(self.moves(), [])

    def test_still_palm_pauses_and_fist_resumes(self):
        events = []
        for _ in range(40):
            events += self.feed(hand(pose="palm"))
        self.assertEqual(events, ["palm_hold"])
        events = []
        for _ in range(40):
            events += self.feed(hand(pose="fist"), active=False)
        self.assertEqual(events, ["fist_hold"])

    def test_two_finger_scroll(self):
        self.feed(hand(y=300, pose="two"))
        self.feed(hand(y=252, pose="two"))  # hand up 1/10 of the frame -> 4 notches up
        self.assertEqual(self.sender.sent, [("scroll", 4)])

    def test_two_hand_zoom(self):
        events = []
        for gap in (300, 380, 480):  # each step widens the pinch distance by more than 1.25x
            events += self.feed(hand(150, pinch="index"), hand(150 + gap, pinch="index"))
        self.assertEqual(events, ["zoom_in", "zoom_in"])

    def test_laser_maps_fingertip_to_screen(self):
        self.hc.laser = True
        lm = hand()
        lm[INDEX_TIP] = (320, 240)
        self.feed(lm)
        self.assertEqual(self.hc.laser_point, (0.5, 0.5))
        self.assertEqual(self.sender.sent, [])


class HeadScrollTest(unittest.TestCase):
    def test_scroll_direction_and_deadzone(self):
        sender = FakeSender()
        hs = HeadScroller(Config(head_scroll_deadzone=0.03, head_scroll_range=0.08, head_scroll_speed=10), sender)
        for i in range(31):
            hs.update(face(pitch=0.52), BASE, i / 30)  # inside the dead zone
        self.assertEqual(sender.sent, [])
        for i in range(31):
            hs.update(face(pitch=0.61), BASE, 2 + i / 30)  # looking down at full speed for 1 s
        self.assertEqual(sum(n for _, n in sender.sent), -10)


class GazeTest(unittest.TestCase):
    def test_calibration_fits_a_model(self):
        cal = GazeCalibration()
        t = 0.0
        while not cal.finished(t):
            state = cal.current(t)
            _, (px, py), _ = state
            # Simulated eyes: iris position is a linear function of the screen point.
            cal.update(face(gaze_x=0.3 + 0.4 * px, gaze_y=-0.1 + 0.2 * py), t)
            t += 1 / 30
        model = cal.fit()
        x, y = model.predict(face(gaze_x=0.5, gaze_y=0.0))
        self.assertAlmostEqual(x, 0.5, places=2)
        self.assertAlmostEqual(y, 0.5, places=2)

    def test_model_roundtrip_and_dwell_click(self):
        home = isolated_home(self)
        model = GazeModel(np.zeros((8, 2)))
        model.coef[0] = (0.25, 0.75)
        model.save()
        loaded = GazeModel.load()
        self.assertEqual(loaded.predict(BASE), (0.25, 0.75))
        self.assertTrue(gaze_model_path().is_relative_to(home))
        pointer = GazePointer(Config(gaze_dwell_click=True, gaze_dwell_time=0.5), loaded)
        clicks = [pointer.update(BASE, i / 10)[1] for i in range(12)]
        self.assertEqual(clicks.count(True), 1)


class VoiceTest(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(words_to_number("twelve".split()), 12)
        self.assertEqual(words_to_number("one hundred and twenty three".split()), 123)
        self.assertEqual(words_to_number(["42"]), 42)
        self.assertIsNone(words_to_number(["banana"]))

    def test_commands(self):
        commands = default_voice_commands()
        self.assertEqual(parse_command("Next Page", commands), "pagedown")
        self.assertEqual(parse_command("go to page twenty one", commands), "@goto:21")
        self.assertEqual(parse_command("profile pdf", commands, ["pdf"]), "@profile:pdf")
        self.assertIsNone(parse_command("profile cooking", commands, ["pdf"]))
        self.assertIsNone(parse_command("hello there", commands))

    def test_grammar_contains_commands_and_numbers(self):
        g = grammar(default_voice_commands(), ["pdf"])
        self.assertIn("next page", g)
        self.assertIn("twenty", g)
        self.assertIn("profile pdf", g)
        self.assertEqual(g[-1], "[unk]")


class SetupWizardTest(unittest.TestCase):
    def test_records_steps_and_derives_thresholds(self):
        poses = {
            "neutral": BASE,
            "wink_left": face(ear_left=0.09, ear_right=0.24),
            "wink_right": face(ear_right=0.09, ear_left=0.24),
            "brow": face(brow=0.65),
            "smile": face(smile=1.4),
            "mouth": face(mar=0.75),
        }
        setup = GuidedSetup()
        t = 0.0
        while not setup.finished:
            setup.update(poses[setup.current()[0]], t)
            t += 1 / 30
        baseline, s = setup.result
        self.assertEqual(baseline, BASE)
        self.assertAlmostEqual(s["close_ratio"], 0.3 + 0.4 * 0.7, places=3)
        self.assertLess(s["close_ratio"], s["open_ratio"])
        self.assertLess(s["open_ratio"], 0.8)  # the other eye squinted to 0.8 during winks
        self.assertAlmostEqual(s["brow_raise_ratio"], 1.18, places=2)
        self.assertGreater(s["mouth_open_threshold"], BASE.mar)
        Config.from_dict(s)  # produces a valid configuration

    def test_clamps_extreme_values(self):
        rec = {k: BASE for k in ("neutral", "wink_left", "wink_right", "brow", "smile", "mouth")}
        _, s = derive_settings(rec)
        Config.from_dict(s)


class KeyboardTest(unittest.TestCase):
    def test_row_then_key_scanning(self):
        kb = ScanKeyboard(interval=1.0)
        kb.tick(0.0)
        kb.tick(1.0)  # row 1: H..N
        self.assertIsNone(kb.select(1.1))
        kb.tick(2.1)  # key I
        self.assertEqual(kb.select(2.2), "i")
        self.assertIsNone(kb.col)
        self.assertEqual(kb.press("⌫"), "backspace")
        self.assertEqual(kb.press("?"), "shift+slash")
        self.assertEqual(kb.typed, "?")

    def test_close_row(self):
        kb = ScanKeyboard()
        kb.row = len(kb.layout) - 1
        self.assertEqual(kb.select(0.0), "@close")


class WellnessTest(unittest.TestCase):
    def test_blink_counting(self):
        b = BlinkMonitor()
        t = 0.0
        for closed in [False, True, True, True, False] + [True] * 30 + [False]:
            b.update(closed, t)
            t += 1 / 30
        self.assertEqual(b.total, 1)  # the long closure is not a blink

    def test_presence(self):
        p = Presence(Config(away_seconds=2.0, pause_on_second_face=True))
        self.assertEqual(p.update(0, 0.0), "")
        self.assertEqual(p.update(0, 2.1), "away")
        self.assertEqual(p.update(1, 2.2), "")
        p.update(2, 3.0)
        self.assertEqual(p.update(2, 4.1), "someone else in view")

    def test_break_reminder(self):
        fb = SilentFeedback()
        w = Wellness(Config(break_reminder_minutes=0.05, low_blink_rate=0), fb)
        for i in range(120):
            w.update(True, False, i * 0.05)
        self.assertEqual(fb.notified, ["Time for a short break"])


class StatsTest(unittest.TestCase):
    def test_save_load_summarize(self):
        isolated_home(self)
        self.assertEqual(summarize(load_sessions()), "No sessions recorded yet.")
        save_session(build_session(0, 600, {"wink_left": 9}, {"left_click": 2}, {"wink_left": 1}, 120, "pdf"))
        text = summarize(load_sessions())
        self.assertIn("wink_left", text)
        self.assertIn("90%", text)
        self.assertIn("12.0 / min", text)


class AutostartTest(unittest.TestCase):
    def test_install_and_remove(self):
        isolated_home(self)
        path = install_autostart(["--profile", "pdf", "--tray"])
        content = path.read_text()
        self.assertIn("--profile pdf --tray", content)
        self.assertEqual(path, autostart_path())
        self.assertTrue(remove_autostart())
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()

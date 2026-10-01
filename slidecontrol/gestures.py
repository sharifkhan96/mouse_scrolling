"""Calibration, face gesture detection and action dispatch (no camera/GUI code)."""

import dataclasses
import logging
import math
from collections import Counter

import numpy as np

from .geometry import FaceFeatures

log = logging.getLogger("slidecontrol")

PAUSE_ACTIONS = {"@pause", "@pause_on", "@resume"}
SCROLL_ACTIONS = {"@scroll_up", "@scroll_down"}
REPEATABLE_ACTIONS = SCROLL_ACTIONS | {"up", "down", "pageup", "pagedown", "ctrl+equal", "ctrl+minus"}
NEXT_KEYS = {"down", "right", "pagedown", "space"}
PREV_KEYS = {"up", "left", "pageup"}


class Calibrator:
    """Collects features of a neutral, eyes-open face and returns their median."""

    def __init__(self, duration, min_samples=15):
        self.duration = duration
        self.min_samples = min_samples
        self.samples = []
        self.start = None

    def add(self, features, now):
        if self.start is None:
            self.start = now
        self.samples.append(dataclasses.astuple(features))

    def progress(self, now):
        if self.start is None:
            return 0.0
        return min(1.0, (now - self.start) / self.duration) if self.duration else 1.0

    def result(self, now):
        """The baseline once enough data is collected, otherwise None."""
        if self.progress(now) < 1.0 or len(self.samples) < self.min_samples:
            return None
        return median_features(self.samples)


def median_features(samples):
    return FaceFeatures(*np.median(np.array(samples), axis=0).tolist())


class HoldTrigger:
    """Fires once when a condition has held for `hold` seconds; re-arms on release.

    With `repeat`, it keeps firing every `repeat` seconds while the condition holds,
    starting `repeat_delay` seconds after the first fire. Releases after more than
    half the hold time without firing are counted as near misses (for stats).
    """

    def __init__(self, hold, repeat=None, repeat_delay=0.0):
        self.hold = hold
        self.repeat = repeat
        self.repeat_delay = repeat_delay
        self.start = None
        self.fired = False
        self.last_fire = None
        self.near_misses = 0
        self._last_seen = None

    def update(self, active, now):
        if not active:
            if self.start is not None and not self.fired and self._last_seen is not None \
                    and self.hold and (self._last_seen - self.start) / self.hold >= 0.5:
                self.near_misses += 1
            self.reset()
            return False
        if self.start is None:
            self.start = now
        self._last_seen = now
        if not self.fired and now - self.start >= self.hold:
            self.fired = True
            self.last_fire = now
            return True
        if self.fired and self.repeat and now - self.last_fire >= self.repeat:
            if now - self.start >= self.hold + self.repeat_delay:
                self.last_fire = now
                return True
        return False

    def reset(self):
        self.start = None
        self.fired = False
        self.last_fire = None
        self._last_seen = None

    def progress(self, now):
        if self.start is None:
            return 0.0
        return 1.0 if self.fired or not self.hold else min(1.0, (now - self.start) / self.hold)


class GestureEngine:
    """Turns a stream of FaceFeatures into discrete gesture events."""

    def __init__(self, cfg, baseline=None):
        self.cfg = cfg
        self.baseline = baseline
        holds = {
            "wink_left": cfg.wink_hold, "wink_right": cfg.wink_hold,
            "long_blink": cfg.long_blink_hold, "mouth_open": cfg.mouth_hold,
            "head_left": cfg.head_turn_hold, "head_right": cfg.head_turn_hold,
            "tilt_left": cfg.tilt_hold, "tilt_right": cfg.tilt_hold,
            "brow_raise": cfg.brow_hold, "smile": cfg.smile_hold,
        }
        self.triggers = {}
        for g in cfg.gestures:
            if cfg.repeat_enabled and cfg.actions.get(g) in REPEATABLE_ACTIONS:
                self.triggers[g] = HoldTrigger(holds[g], cfg.repeat_interval, cfg.repeat_delay)
            else:
                self.triggers[g] = HoldTrigger(holds[g])

    def eye_ratios(self, f):
        return f.ear_left / self.baseline.ear_left, f.ear_right / self.baseline.ear_right

    def conditions(self, f):
        c, b = self.cfg, self.baseline
        left, right = self.eye_ratios(f)
        yaw = f.yaw - b.yaw
        roll = f.roll - b.roll
        return {
            "wink_left": left < c.close_ratio and right > c.open_ratio,
            "wink_right": right < c.close_ratio and left > c.open_ratio,
            "long_blink": left < c.close_ratio and right < c.close_ratio,
            "mouth_open": f.mar > c.mouth_open_threshold,
            "head_left": yaw > c.yaw_threshold,
            "head_right": yaw < -c.yaw_threshold,
            "tilt_left": roll > c.roll_threshold_deg,
            "tilt_right": roll < -c.roll_threshold_deg,
            "brow_raise": b.brow > 0 and f.brow / b.brow > c.brow_raise_ratio,
            "smile": b.smile > 0 and f.smile / b.smile > c.smile_ratio and f.mar < c.mouth_open_threshold,
        }

    def update(self, features, now):
        """Returns the gestures that fired on this frame (face lost = all released)."""
        if features is None or self.baseline is None:
            for t in self.triggers.values():
                t.reset()
            return []
        active = self.conditions(features)
        return [g for g, t in self.triggers.items() if t.update(active[g], now)]

    def progress(self, now):
        return {g: t.progress(now) for g, t in self.triggers.items()}

    def near_misses(self):
        return Counter({g: t.near_misses for g, t in self.triggers.items() if t.near_misses})


class Dispatcher:
    """Maps gestures, hand events and voice commands to actions.

    Applies the pause state and a global cooldown. Key presses and scrolling are
    performed here; other internal "@..." actions are returned to the app.
    """

    def __init__(self, cfg, sender, feedback=None):
        self.cfg = cfg
        self.sender = sender
        self.feedback = feedback
        self.paused = False
        self.auto_paused = ""  # reason, e.g. "away"; empty when not auto-paused
        self.last_action_time = -math.inf
        self.last_action = ""
        self.last_gesture = None
        self.counts = Counter()
        self.position = 0  # net next/previous steps, shown in the presenter overlay

    @property
    def active(self):
        return not self.paused and not self.auto_paused

    def handle(self, gesture, now):
        return self.perform(gesture, self.cfg.actions.get(gesture), now)

    def perform(self, source, action, now, cooldown=True):
        """Performs `action`; returns it if the app must handle it, else None."""
        if not action:
            return None
        if not self.active and action not in PAUSE_ACTIONS:
            return None
        repeating = action in REPEATABLE_ACTIONS and source == self.last_gesture
        if cooldown and now - self.last_action_time < self.cfg.cooldown and not repeating:
            log.debug("Ignored %s (cooldown)", source)
            return None
        self.last_action_time = now
        self.last_gesture = source
        self.last_action = f"{source} -> {action}"
        self.counts[source] += 1
        log.info("%s -> %s", source, action)

        if action in PAUSE_ACTIONS:
            self.set_paused(not self.paused if action == "@pause" else action == "@pause_on")
            return action
        if self.feedback and not repeating:
            self.feedback.play("action")
        if action in SCROLL_ACTIONS:
            amount = self.cfg.scroll_amount
            self.sender.scroll(amount if action == "@scroll_up" else -amount)
            return None
        if action.startswith("@goto:"):
            self.goto(action.split(":", 1)[1])
            return None
        if action.startswith("@"):
            return action
        self.sender.send(action)
        if action in NEXT_KEYS:
            self.position += 1
        elif action in PREV_KEYS:
            self.position -= 1
        return None

    def set_paused(self, paused):
        if paused != self.paused:
            self.paused = paused
            log.info("Control %s", "paused" if paused else "resumed")
            if self.feedback:
                self.feedback.play("pause" if paused else "resume")

    def goto(self, number):
        if not number.isdigit():
            return
        if self.cfg.goto_prefix:
            self.sender.send(self.cfg.goto_prefix)
        for digit in number:
            self.sender.send(digit)
        self.sender.send("enter")

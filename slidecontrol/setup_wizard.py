"""Guided setup: records your own gestures and derives thresholds from them."""

import dataclasses

from .gestures import median_features

STEPS = [
    ("neutral", "Look at the screen and relax, both eyes open", 3.0),
    ("wink_left", "Close your LEFT eye, keep the right one open", 2.5),
    ("wink_right", "Close your RIGHT eye, keep the left one open", 2.5),
    ("brow", "Raise your eyebrows", 2.5),
    ("smile", "Smile widely (mouth closed)", 2.5),
    ("mouth", "Open your mouth wide", 2.5),
]
SETTLE = 0.8  # seconds to get into position before recording each step


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def derive_settings(rec):
    """Thresholds from the median features of each step. Returns (baseline, settings)."""
    base = rec["neutral"]
    wl, wr = rec["wink_left"], rec["wink_right"]
    closed = max(wl.ear_left / base.ear_left, wr.ear_right / base.ear_right)
    other_open = min(wl.ear_right / base.ear_right, wr.ear_left / base.ear_left)
    close_ratio = _clamp(closed + 0.4 * (1 - closed), 0.3, 0.8)
    open_ratio = _clamp(min(other_open * 0.85, 0.95), close_ratio + 0.05, 0.95)

    brow = rec["brow"].brow / base.brow if base.brow else 1.3
    smile = rec["smile"].smile / base.smile if base.smile else 1.3
    mouth = rec["mouth"].mar
    settings = {
        "close_ratio": round(close_ratio, 3),
        "open_ratio": round(open_ratio, 3),
        "brow_raise_ratio": round(_clamp(1 + 0.6 * (brow - 1), 1.05, 1.6), 3),
        "smile_ratio": round(_clamp(1 + 0.6 * (smile - 1), 1.05, 1.6), 3),
        "mouth_open_threshold": round(_clamp(base.mar + 0.6 * (mouth - base.mar), 0.2, 0.9), 3),
    }
    return base, settings


class GuidedSetup:
    def __init__(self, steps=STEPS):
        self.steps = steps
        self.index = 0
        self.step_start = None
        self.samples = []
        self.recorded = {}
        self.result = None  # (baseline, settings) when finished

    @property
    def finished(self):
        return self.index >= len(self.steps)

    def current(self):
        return None if self.finished else self.steps[self.index]

    def progress(self, now):
        if self.finished or self.step_start is None:
            return 0.0
        return min(1.0, (now - self.step_start) / (SETTLE + self.steps[self.index][2]))

    def update(self, features, now):
        if self.finished:
            return
        if self.step_start is None:
            self.step_start = now
        key, _, duration = self.steps[self.index]
        elapsed = now - self.step_start
        if features is not None and elapsed >= SETTLE:
            self.samples.append(dataclasses.astuple(features))
        if elapsed >= SETTLE + duration:
            if len(self.samples) < 5:  # face lost: repeat the step
                self.step_start, self.samples = now, []
                return
            self.recorded[key] = median_features(self.samples)
            self.index += 1
            self.step_start, self.samples = now, []
            if self.finished:
                self.result = derive_settings(self.recorded)

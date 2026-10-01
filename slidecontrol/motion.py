"""Continuous controls: head-tilt scrolling and gaze-driven pointer."""

import json
import math

import numpy as np

from .config import data_dir


class HeadScroller:
    """Look down to scroll down, look up to scroll up; speed grows with the tilt."""

    def __init__(self, cfg, sender):
        self.cfg = cfg
        self.sender = sender
        self.remainder = 0.0
        self.last = None
        self.speed = 0.0  # signed notches/second, for the HUD

    def update(self, features, baseline, now, active=True):
        dt = 0.0 if self.last is None else min(now - self.last, 0.1)
        self.last = now
        if not active or features is None or baseline is None:
            self.remainder = self.speed = 0.0
            return
        c = self.cfg
        deviation = features.pitch - baseline.pitch
        excess = abs(deviation) - c.head_scroll_deadzone
        if excess <= 0:
            self.remainder = self.speed = 0.0
            return
        # Looking down increases pitch and scrolls down (negative wheel).
        self.speed = -math.copysign(min(excess / c.head_scroll_range, 1.0) * c.head_scroll_speed, deviation)
        self.remainder += self.speed * dt
        notches = int(self.remainder)
        if notches:
            self.remainder -= notches
            self.sender.scroll(notches)


# ---------------------------------------------------------------- gaze

CALIBRATION_POINTS = [(x, y) for y in (0.1, 0.5, 0.9) for x in (0.1, 0.5, 0.9)]
POINT_SETTLE = 0.6  # seconds to move the eyes to a new dot before sampling
POINT_SAMPLE = 1.0


def gaze_design(features):
    ex, ey = features.gaze_x, features.gaze_y
    return [1.0, ex, ey, ex * ey, ex * ex, ey * ey, features.yaw, features.pitch]


class GazeModel:
    """Ridge regression from eye/head features to normalized screen coordinates."""

    def __init__(self, coef=None):
        self.coef = None if coef is None else np.asarray(coef)

    def fit(self, features, targets, alpha=1e-3):
        X = np.array([gaze_design(f) for f in features])
        Y = np.array(targets)
        reg = alpha * np.eye(X.shape[1])
        reg[0, 0] = 0  # don't penalize the intercept
        self.coef = np.linalg.solve(X.T @ X + reg, X.T @ Y)
        return self

    def predict(self, features):
        x, y = np.array(gaze_design(features)) @ self.coef
        return float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1))

    def save(self, path=None):
        path = path or gaze_model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"coef": self.coef.tolist()}))

    @classmethod
    def load(cls, path=None):
        path = path or gaze_model_path()
        if not path.is_file():
            return None
        return cls(json.loads(path.read_text())["coef"])


def gaze_model_path():
    return data_dir() / "gaze.json"


class GazeCalibration:
    """Shows dots one by one and records eye features while you look at each."""

    def __init__(self, points=CALIBRATION_POINTS):
        self.points = points
        self.start = None
        self.features, self.targets = [], []

    def current(self, now):
        """(index, point, sampling) of the dot to show, or None when finished."""
        if self.start is None:
            self.start = now
        index = int((now - self.start) // (POINT_SETTLE + POINT_SAMPLE))
        if index >= len(self.points):
            return None
        sampling = (now - self.start) % (POINT_SETTLE + POINT_SAMPLE) >= POINT_SETTLE
        return index, self.points[index], sampling

    def update(self, features, now):
        state = self.current(now)
        if state is None or features is None:
            return
        _, point, sampling = state
        if sampling:
            self.features.append(features)
            self.targets.append(point)

    def finished(self, now):
        return self.current(now) is None

    def fit(self):
        if len({t for t in self.targets}) < len(self.points) - 1:
            return None  # face was lost for too many dots
        return GazeModel().fit(self.features, self.targets)


class GazePointer:
    """Smooths gaze predictions and optionally clicks when you dwell on one spot."""

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.point = None
        self.dwell_anchor = None
        self.dwell_start = None

    def update(self, features, now):
        """Returns (point, dwell_click)."""
        if features is None or self.model is None:
            self.point = None
            return None, False
        target = np.array(self.model.predict(features))
        a = self.cfg.gaze_smoothing
        self.point = target if self.point is None else a * self.point + (1 - a) * target
        click = False
        if self.cfg.gaze_dwell_click:
            if self.dwell_anchor is None or np.linalg.norm(self.point - self.dwell_anchor) > self.cfg.gaze_dwell_radius:
                self.dwell_anchor, self.dwell_start = self.point.copy(), now
            elif now - self.dwell_start >= self.cfg.gaze_dwell_time:
                click = True
                self.dwell_start = math.inf  # one click per dwell
        return (float(self.point[0]), float(self.point[1])), click

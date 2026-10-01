"""Hand tracking: pointer, clicks, swipes, two-finger scroll, two-hand zoom, laser pointer."""

import math
from collections import Counter, deque

import numpy as np

from .geometry import dist

# MediaPipe Hands landmark indices.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20
FINGERS = [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP), (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]

SWIPE_COOLDOWN = 0.8
STILL_TOLERANCE = 0.05  # fraction of frame width the palm may drift while "held still"


def fingers_extended(lm):
    """Index, middle, ring, pinky: a finger is extended if its tip is farther from the wrist than its middle joint."""
    return [dist(lm[tip], lm[WRIST]) > dist(lm[pip], lm[WRIST]) for tip, pip in FINGERS]


def hand_size(lm):
    return dist(lm[WRIST], lm[MIDDLE_MCP]) or 1.0


def is_pinching(lm, threshold):
    """Thumb and index tips touching, with the index reaching out (so a fist doesn't count)."""
    size = hand_size(lm)
    reaching = dist(lm[INDEX_TIP], lm[WRIST]) > 1.15 * dist(lm[INDEX_MCP], lm[WRIST])
    return reaching and dist(lm[THUMB_TIP], lm[INDEX_TIP]) / size < threshold


def hand_pose(lm):
    ext = fingers_extended(lm)
    if all(ext):
        return "palm"
    if not any(ext):
        return "fist"
    if ext == [True, True, False, False]:
        return "two"
    return "point"


class HandController:
    """Turns hand landmarks into pointer movement, clicks, scrolling and hand events.

    Poses (one hand):
      point / pinch : move the pointer like an air trackpad; thumb+index pinch = left
                      button (hold to drag); thumb+middle pinch = right click
      two fingers   : (index + middle up) move the hand up/down to scroll
      open palm     : swipe sideways for swipe_left/right; hold still for palm_hold
      fist          : clutch, i.e. reposition your hand without moving the pointer;
                      hold it for fist_hold
    Two hands pinching: pull apart / push together for zoom_in / zoom_out.
    Laser mode: the index fingertip drives an on-screen dot instead of the pointer.
    """

    def __init__(self, cfg, sender):
        self.cfg = cfg
        self.sender = sender
        self.enabled = cfg.hand_control
        self.pointer_enabled = True  # turned off while gaze drives the pointer
        self.laser = False
        self.laser_point = None
        self.left_down = False
        self.right_pinched = False
        self.clicks = Counter()
        self.pose = None
        self._pose_start = None
        self._pose_anchor = None
        self._pose_fired = False
        self._last_swipe = -math.inf
        self.reset()

    def reset(self):
        """Forgets the hand and releases a held button."""
        self._release()
        self.right_pinched = False
        self.prev = None
        self.velocity = np.zeros(2)
        self.remainder = np.zeros(2)
        self.scroll_remainder = 0.0
        self.history = deque()
        self.zoom_ref = None
        self.index_pinch = self.middle_pinch = math.inf
        self.pose = None
        self.laser_point = None
        self.hands_seen = 0

    def _release(self):
        if self.left_down:
            self.sender.button("left", False)
        self.left_down = False

    def update(self, hands, frame_w, frame_h, now, active=True):
        """`hands` is a list of (21, 2) landmark arrays in pixels. Returns hand events."""
        if not self.enabled or not hands:
            self.reset()
            return []
        self.hands_seen = len(hands)
        events = []
        if len(hands) >= 2 and active and not self.laser:
            self._release()
            self.prev = None
            self.pose = "two hands"
            return self._zoom(hands)
        self.zoom_ref = None

        lm = hands[0]
        size = hand_size(lm)
        self.index_pinch = dist(lm[THUMB_TIP], lm[INDEX_TIP]) / size
        self.middle_pinch = dist(lm[THUMB_TIP], lm[MIDDLE_TIP]) / size
        pinching = self.left_down or is_pinching(lm, self.cfg.pinch_threshold)
        pose = "pinch" if pinching else hand_pose(lm)
        point = lm[MIDDLE_MCP]
        events += self._pose_hold(pose, point, frame_w, now, active)
        self.pose = pose

        if not active:
            self._release()
            self.prev = None
            return events
        if self.laser:
            self._release()
            self.prev = None
            self.laser_point = self._laser(lm[INDEX_TIP], frame_w, frame_h)
            return events
        self.laser_point = None

        self.history.append((now, point[0] / frame_w))
        while self.history and now - self.history[0][0] > self.cfg.swipe_time:
            self.history.popleft()

        if pose == "palm":
            self._release()
            self.prev = None
            events += self._swipe(now)
        elif pose == "fist":
            self._release()
            self.prev = None
        elif pose == "two":
            self._release()
            self._scroll(point, frame_h)
            self.prev = point
        else:
            self._pointer(point, frame_w)
            self._clicks(pinching)
            self.prev = point
        return events

    def _pose_hold(self, pose, point, frame_w, now, active):
        if pose != self.pose:
            self._pose_start, self._pose_anchor, self._pose_fired = now, point, False
            return []
        if self._pose_fired or now - self._pose_start < self.cfg.pose_hold:
            return []
        if pose == "palm" and active and dist(point, self._pose_anchor) / frame_w < STILL_TOLERANCE:
            self._pose_fired = True
            return ["palm_hold"]
        if pose == "fist" and not active:
            self._pose_fired = True
            return ["fist_hold"]
        return []

    def _swipe(self, now):
        if len(self.history) < 2 or now - self._last_swipe < SWIPE_COOLDOWN:
            return []
        dx = self.history[-1][1] - self.history[0][1]
        if abs(dx) < self.cfg.swipe_distance:
            return []
        self._last_swipe = now
        self.history.clear()
        # The camera image is mirrored: moving your hand to your left moves it right in the image.
        return ["swipe_left" if dx > 0 else "swipe_right"]

    def _scroll(self, point, frame_h):
        if self.prev is None:
            return
        # Hand up (smaller y) scrolls up (positive wheel).
        self.scroll_remainder += -(point[1] - self.prev[1]) / frame_h * self.cfg.hand_scroll_speed
        notches = int(self.scroll_remainder)
        if notches:
            self.scroll_remainder -= notches
            self.sender.scroll(notches)

    def _pointer(self, point, frame_w):
        if self.prev is None or not self.pointer_enabled:
            return
        c = self.cfg
        # The camera sees you mirrored: moving your hand right moves it left in the image.
        delta = (point - self.prev) / frame_w * c.pointer_speed * np.array([-1.0, 1.0])
        a = c.pointer_smoothing
        self.velocity = a * self.velocity + (1 - a) * delta
        if np.linalg.norm(self.velocity) >= c.pointer_deadzone:
            self.remainder += self.velocity
            step = np.trunc(self.remainder)
            self.remainder -= step
            if step.any():
                self.sender.move(int(step[0]), int(step[1]))

    def _clicks(self, pinching):
        c = self.cfg
        if not self.left_down and pinching:
            self.left_down = True
            self.clicks["left_click"] += 1
            self.sender.button("left", True)
        elif self.left_down and self.index_pinch > c.pinch_release:
            self._release()

        if not self.right_pinched and not self.left_down and self.middle_pinch < c.pinch_threshold:
            self.right_pinched = True
            self.clicks["right_click"] += 1
            self.sender.click("right")
        elif self.right_pinched and self.middle_pinch > c.pinch_release:
            self.right_pinched = False

    def _zoom(self, hands):
        a, b = hands[0], hands[1]
        if not (is_pinching(a, self.cfg.pinch_threshold) and is_pinching(b, self.cfg.pinch_threshold)):
            self.zoom_ref = None
            return []
        mid_a = (a[THUMB_TIP] + a[INDEX_TIP]) / 2
        mid_b = (b[THUMB_TIP] + b[INDEX_TIP]) / 2
        d = dist(mid_a, mid_b)
        if self.zoom_ref is None:
            self.zoom_ref = d
            return []
        if d / self.zoom_ref >= self.cfg.zoom_step:
            self.zoom_ref = d
            return ["zoom_in"]
        if d / self.zoom_ref <= 1 / self.cfg.zoom_step:
            self.zoom_ref = d
            return ["zoom_out"]
        return []

    def _laser(self, tip, frame_w, frame_h):
        r = self.cfg.laser_region
        x = 1 - tip[0] / frame_w  # mirrored so the dot follows your hand
        y = tip[1] / frame_h
        clip = lambda v: min(1.0, max(0.0, (v - (1 - r) / 2) / r))  # noqa: E731
        return clip(x), clip(y)

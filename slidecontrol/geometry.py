"""Face measurements computed from MediaPipe FaceMesh landmarks."""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

# MediaPipe FaceMesh landmark indices. "Right"/"left" are the subject's own
# sides (the right eye appears on the left of an unmirrored camera image).
RIGHT_EYE = [33, 160, 158, 133, 153, 144]  # p1..p6 for the EAR formula
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]  # inner top, inner bottom, left corner, right corner
MOUTH_LEFT, MOUTH_RIGHT = 61, 291  # outer mouth corners
NOSE_TIP = 1
RIGHT_CHEEK = 234
LEFT_CHEEK = 454
FOREHEAD = 10
CHIN = 152
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 33, 133
LEFT_EYE_INNER, LEFT_EYE_OUTER = 362, 263
RIGHT_BROW, LEFT_BROW = 105, 334
RIGHT_IRIS, LEFT_IRIS = 468, 473  # only present with refine_landmarks=True


@dataclass
class FaceFeatures:
    ear_left: float
    ear_right: float
    mar: float
    yaw: float    # 0.5 = nose centred between cheeks; grows when turning to your left
    roll: float   # degrees; grows when tilting toward your left shoulder
    pitch: float  # nose height between forehead and chin; grows when looking down
    brow: float   # brow height above the eye corners / eye distance
    smile: float  # mouth width / eye distance
    gaze_x: float  # iris position within the eyes, 0 = image left, 1 = image right
    gaze_y: float  # iris offset from the eye-corner line, relative to eye width

    def blend(self, prev, alpha):
        """Exponential moving average with the previous (smoothed) features."""
        if prev is None or alpha == 0:
            return self
        return FaceFeatures(*(alpha * p + (1 - alpha) * c
                              for p, c in zip(dataclasses.astuple(prev), dataclasses.astuple(self))))


def dist(a, b):
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts):
    """EAR from six points p1..p6 (Soukupova & Cech, 2016)."""
    p1, p2, p3, p4, p5, p6 = pts
    horizontal = dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * horizontal)


def mouth_aspect_ratio(pts):
    top, bottom, left, right = pts
    width = dist(left, right)
    return dist(top, bottom) / width if width else 0.0


def _iris_offset(lm, iris, corner_a, corner_b):
    a, b = lm[corner_a], lm[corner_b]
    width = b[0] - a[0]
    if not width:
        return 0.5, 0.0
    x = (lm[iris, 0] - a[0]) / width
    y = (lm[iris, 1] - (a[1] + b[1]) / 2) / width
    return x, y


def extract_features(lm):
    """Compute FaceFeatures from an (N, 2) array of landmarks in pixel units."""
    span = lm[LEFT_CHEEK, 0] - lm[RIGHT_CHEEK, 0]
    yaw = (lm[NOSE_TIP, 0] - lm[RIGHT_CHEEK, 0]) / span if span else 0.5
    dx, dy = lm[LEFT_EYE_OUTER] - lm[RIGHT_EYE_OUTER]
    roll = math.degrees(math.atan2(dy, dx))
    face_height = lm[CHIN, 1] - lm[FOREHEAD, 1]
    pitch = (lm[NOSE_TIP, 1] - lm[FOREHEAD, 1]) / face_height if face_height else 0.5
    eye_distance = dist(lm[RIGHT_EYE_OUTER], lm[LEFT_EYE_OUTER]) or 1.0
    # Measured from the eye corners, not the lids, so blinking doesn't look like a brow raise.
    right_eye_mid = (lm[RIGHT_EYE_OUTER, 1] + lm[RIGHT_EYE_INNER, 1]) / 2
    left_eye_mid = (lm[LEFT_EYE_OUTER, 1] + lm[LEFT_EYE_INNER, 1]) / 2
    brow = ((right_eye_mid - lm[RIGHT_BROW, 1]) + (left_eye_mid - lm[LEFT_BROW, 1])) / 2 / eye_distance
    smile = dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT]) / eye_distance
    if len(lm) > LEFT_IRIS:
        rx, ry = _iris_offset(lm, RIGHT_IRIS, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
        lx, ly = _iris_offset(lm, LEFT_IRIS, LEFT_EYE_INNER, LEFT_EYE_OUTER)
        gaze_x, gaze_y = (rx + lx) / 2, (ry + ly) / 2
    else:
        gaze_x, gaze_y = 0.5, 0.0
    return FaceFeatures(
        ear_left=eye_aspect_ratio(lm[LEFT_EYE]),
        ear_right=eye_aspect_ratio(lm[RIGHT_EYE]),
        mar=mouth_aspect_ratio(lm[MOUTH]),
        yaw=float(yaw),
        roll=float(roll),
        pitch=float(pitch),
        brow=float(brow),
        smile=float(smile),
        gaze_x=float(gaze_x),
        gaze_y=float(gaze_y),
    )


def face_size(lm):
    """Rough face size in pixels, used to pick the main face when several are seen."""
    return dist(lm[RIGHT_CHEEK], lm[LEFT_CHEEK])

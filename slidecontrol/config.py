"""Settings, profiles and file locations."""

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROFILES_DIR = PROJECT_DIR / "profiles"
DEFAULT_VOICE_MODEL = PROJECT_DIR / "models" / "vosk-model-small-en-us-0.15"


def config_dir():
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "face-slide-control"


def data_dir():
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "face-slide-control"


def user_config_path():
    return config_dir() / "config.json"


FACE_GESTURES = [
    "wink_left", "wink_right", "long_blink", "mouth_open",
    "head_left", "head_right", "tilt_left", "tilt_right",
    "brow_raise", "smile",
]
HAND_EVENTS = ["swipe_left", "swipe_right", "palm_hold", "fist_hold", "zoom_in", "zoom_out"]

# Merged key-by-key (instead of replaced) when configs and profiles are layered.
DICT_FIELDS = ("actions", "voice_commands")


def default_actions():
    return {
        "wink_left": "down",
        "wink_right": "up",
        "long_blink": "@pause",
        "mouth_open": "b",
        "head_left": "up",
        "head_right": "down",
        "tilt_left": "up",
        "tilt_right": "down",
        "brow_raise": "pagedown",
        "smile": "shift+print",
        "swipe_left": "pagedown",
        "swipe_right": "pageup",
        "palm_hold": "@pause_on",
        "fist_hold": "@resume",
        "zoom_in": "ctrl+equal",
        "zoom_out": "ctrl+minus",
    }


def default_voice_commands():
    return {
        "next": "pagedown", "next page": "pagedown", "next slide": "pagedown",
        "back": "pageup", "previous": "pageup", "previous page": "pageup", "previous slide": "pageup",
        "down": "down", "up": "up", "scroll down": "@scroll_down", "scroll up": "@scroll_up",
        "first page": "home", "last page": "end",
        "zoom in": "ctrl+equal", "zoom out": "ctrl+minus", "full screen": "f11",
        "screenshot": "shift+print",
        "stop": "@pause_on", "pause": "@pause_on", "resume": "@resume", "start": "@resume",
        "laser": "@laser", "keyboard": "@keyboard", "overlay": "@overlay",
        "calibrate": "@recalibrate", "settings": "@settings", "setup": "@setup",
    }


@dataclass
class Config:
    # Camera and display
    camera: int = 0
    width: int = 640
    height: int = 480
    mirror: bool = True
    show_preview: bool = True
    show_mesh: bool = True
    overlay: bool = False  # presenter overlay window
    dry_run: bool = False
    input_backend: str = "auto"  # "auto", "uinput" or "pyautogui"
    profile: str = ""

    # Face gestures
    calibration_seconds: float = 3.0
    smoothing: float = 0.5  # EMA factor for features, 0 = none, closer to 1 = smoother
    # Eye thresholds are relative to the calibrated open-eye EAR.
    close_ratio: float = 0.55
    open_ratio: float = 0.65  # the other eye may squint a little while winking
    wink_hold: float = 0.2
    long_blink_hold: float = 1.2
    mouth_open_threshold: float = 0.55  # absolute mouth aspect ratio
    mouth_hold: float = 0.6
    yaw_threshold: float = 0.12  # shift of nose position across the face width
    head_turn_hold: float = 0.35
    roll_threshold_deg: float = 15.0
    tilt_hold: float = 0.35
    brow_raise_ratio: float = 1.15  # brow height relative to calibration
    brow_hold: float = 0.4
    smile_ratio: float = 1.2  # mouth width relative to calibration
    smile_hold: float = 0.6
    cooldown: float = 0.6  # minimum seconds between any two actions

    # Scrolling actions (arrow/page keys, @scroll_*) repeat while held, like key auto-repeat.
    repeat_enabled: bool = True
    repeat_delay: float = 0.5
    repeat_interval: float = 0.12
    scroll_amount: int = 3  # wheel notches per @scroll_* step
    goto_prefix: str = ""  # keys pressed before typing a page number ("ctrl+l" for Evince)

    # Head-tilt scrolling: look up/down to scroll with the mouse wheel.
    head_scroll: bool = False
    head_scroll_deadzone: float = 0.03
    head_scroll_range: float = 0.08  # deflection beyond the dead zone for full speed
    head_scroll_speed: float = 15.0  # wheel notches per second at full speed

    # Hand tracking
    hand_control: bool = True
    pointer_speed: float = 2000.0  # pointer pixels per full camera width of hand movement
    pointer_smoothing: float = 0.5
    pointer_deadzone: float = 2.0
    pinch_threshold: float = 0.25  # relative to hand size (wrist to middle knuckle)
    pinch_release: float = 0.4
    swipe_distance: float = 0.25  # fraction of the frame width
    swipe_time: float = 0.35
    pose_hold: float = 1.0  # seconds to hold an open palm / fist
    hand_scroll_speed: float = 40.0  # wheel notches per frame height of two-finger movement
    zoom_step: float = 1.25  # change in two-hand distance per zoom step
    laser_region: float = 0.6  # central part of the camera image mapped to the whole screen

    # Gaze tracking
    gaze: bool = False
    gaze_smoothing: float = 0.85
    gaze_dwell_click: bool = False
    gaze_dwell_time: float = 1.2
    gaze_dwell_radius: float = 0.03

    # Voice commands
    voice: bool = False
    voice_model: str = ""

    # Presence and wellness
    auto_pause_away: bool = True
    away_seconds: float = 3.0
    pause_on_second_face: bool = False
    break_reminder_minutes: float = 20.0  # 0 = off
    low_blink_rate: float = 8.0  # blinks per minute; 0 = off

    # Feedback
    sounds: bool = True
    notifications: bool = True

    gestures: list = field(default_factory=lambda: ["wink_left", "wink_right", "long_blink"])
    actions: dict = field(default_factory=default_actions)
    voice_commands: dict = field(default_factory=default_voice_commands)

    @classmethod
    def from_dict(cls, data):
        data = {k: v for k, v in data.items() if not k.startswith("_")}  # "_description" etc.
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
        cfg = cls(**{k: v for k, v in data.items() if k not in DICT_FIELDS})
        for name in DICT_FIELDS:
            getattr(cfg, name).update(data.get(name, {}))
        cfg.validate()
        return cfg

    def to_dict(self):
        return dataclasses.asdict(self)

    def validate(self):
        bad = [g for g in self.gestures if g not in FACE_GESTURES]
        if bad:
            raise ValueError(f"Unknown gestures: {', '.join(bad)} (choose from {', '.join(FACE_GESTURES)})")
        if not 0 <= self.smoothing < 1 or not 0 <= self.gaze_smoothing < 1:
            raise ValueError("smoothing values must be in [0, 1)")
        if self.close_ratio >= self.open_ratio:
            raise ValueError("close_ratio must be lower than open_ratio")
        if self.input_backend not in ("auto", "uinput", "pyautogui"):
            raise ValueError("input_backend must be 'auto', 'uinput' or 'pyautogui'")
        if self.pinch_threshold >= self.pinch_release:
            raise ValueError("pinch_threshold must be lower than pinch_release")
        if self.zoom_step <= 1:
            raise ValueError("zoom_step must be greater than 1")

    @property
    def voice_model_path(self):
        return Path(self.voice_model) if self.voice_model else DEFAULT_VOICE_MODEL


def merge(base, override):
    """Layers config dicts; action and voice-command maps are merged key by key."""
    out = dict(base)
    for k, v in override.items():
        if k in DICT_FIELDS:
            out[k] = {**base.get(k, {}), **v}
        else:
            out[k] = v
    return out


def profile_dirs():
    return [config_dir() / "profiles", PROFILES_DIR]


def list_profiles():
    names = set()
    for d in profile_dirs():
        if d.is_dir():
            names.update(p.stem for p in d.glob("*.json"))
    return sorted(names)


def load_profile(name):
    for d in profile_dirs():
        path = d / f"{name}.json"
        if path.is_file():
            with open(path) as fh:
                return json.load(fh)
    raise ValueError(f"Unknown profile {name!r} (available: {', '.join(list_profiles()) or 'none'})")


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(path)

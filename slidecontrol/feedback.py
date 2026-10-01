"""Sound cues and desktop notifications."""

import logging
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger("slidecontrol")

SOUND_DIR = Path("/usr/share/sounds/freedesktop/stereo")
SOUNDS = {
    "action": "audio-volume-change.oga",
    "pause": "device-removed.oga",
    "resume": "device-added.oga",
    "calibrated": "complete.oga",
    "reminder": "message-new-instant.oga",
    "screenshot": "camera-shutter.oga",
    "toggle": "dialog-information.oga",
}
MIN_GAP = 0.15  # seconds between two plays of the same sound


class Feedback:
    def __init__(self, cfg):
        self.cfg = cfg
        self.player = shutil.which("pw-play") or shutil.which("paplay")
        self.notifier = shutil.which("notify-send")
        self._last = {}

    def play(self, event):
        if not self.cfg.sounds or not self.player:
            return
        path = SOUND_DIR / SOUNDS.get(event, "")
        now = time.monotonic()
        if not path.is_file() or now - self._last.get(event, -1) < MIN_GAP:
            return
        self._last[event] = now
        self._spawn([self.player, str(path)])

    def notify(self, title, body="", urgent=False):
        log.info("%s %s", title, body)
        if not self.cfg.notifications or not self.notifier:
            return
        cmd = [self.notifier, "-a", "Face Slide Control", "-i", "camera-web"]
        if urgent:
            cmd += ["-u", "critical"]
        self._spawn(cmd + [title, body])

    @staticmethod
    def _spawn(cmd):
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            log.debug("Could not run %s: %s", cmd[0], exc)

"""Desktop integration: tray icon process and start-at-login."""

import logging
import os
import queue
import shlex
import subprocess
import sys
import threading
from pathlib import Path

from .config import PROJECT_DIR

log = logging.getLogger("slidecontrol")

SYSTEM_PYTHONS = ["/usr/bin/python3", "/usr/local/bin/python3"]


class Tray:
    """Runs tray_helper.py with a Python that has GTK bindings and relays menu commands."""

    def __init__(self):
        self.commands = queue.Queue()
        self.proc = None
        self._status = None

    def start(self):
        helper = Path(__file__).with_name("tray_helper.py")
        for python in SYSTEM_PYTHONS + [sys.executable]:
            if not os.path.exists(python):
                continue
            check = subprocess.run([python, "-c", "import gi; gi.require_version('Gtk', '3.0')"],
                                   capture_output=True)
            if check.returncode == 0:
                break
        else:
            log.warning("Tray icon unavailable: no Python with GTK bindings (install python3-gi)")
            return False
        # GTK's legacy status icon needs X11; XWayland provides it on Wayland desktops.
        env = dict(os.environ, GDK_BACKEND="x11")
        self.proc = subprocess.Popen([python, str(helper)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
        threading.Thread(target=self._read, daemon=True).start()
        log.info("Tray icon started")
        return True

    def _read(self):
        for line in self.proc.stdout:
            self.commands.put(line.strip())

    def set_status(self, text):
        if self.proc and self.proc.poll() is None and text != self._status:
            self._status = text
            try:
                self.proc.stdin.write(text + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()


def autostart_path():
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "autostart" / "face-slide-control.desktop"


def install_autostart(args):
    """Starts the app at login with the given command-line arguments."""
    launcher = PROJECT_DIR / "face_slide_control.py"
    command = " ".join(shlex.quote(a) for a in [sys.executable, str(launcher), *args])
    path = autostart_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Face Slide Control\n"
        "Comment=Hands-free control with face, hand and voice gestures\n"
        f"Exec={command}\n"
        f"Path={PROJECT_DIR}\n"
        "Icon=camera-web\n"
        "Terminal=false\n"
        "X-GNOME-Autostart-enabled=true\n"
        "X-GNOME-Autostart-Delay=5\n"
    )
    return path


def remove_autostart():
    path = autostart_path()
    if path.exists():
        path.unlink()
        return True
    return False


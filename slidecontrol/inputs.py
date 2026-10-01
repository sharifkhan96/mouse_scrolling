"""Sending keys and mouse events: Linux uinput (Wayland + X11) or pyautogui fallback."""

import logging

log = logging.getLogger("slidecontrol")

KEY_ALIASES = {
    "ctrl": "leftctrl", "control": "leftctrl", "shift": "leftshift", "alt": "leftalt",
    "win": "leftmeta", "super": "leftmeta", "cmd": "leftmeta", "escape": "esc",
    "return": "enter", "pgdn": "pagedown", "pgup": "pageup", "del": "delete", " ": "space",
    "print": "sysrq", "printscreen": "sysrq", "plus": "equal", "=": "equal", "-": "minus",
    ".": "dot", ",": "comma", "/": "slash", ";": "semicolon", "'": "apostrophe",
}
ABS_MAX = 32767


class UInputBackend:
    """Virtual keyboard and mice via /dev/uinput; works on Wayland and X11."""

    def __init__(self):
        from evdev import UInput, ecodes
        self.e = ecodes
        self._UInput = UInput
        self.keyboard = UInput({ecodes.EV_KEY: list(range(1, 249))}, name="face-slide-control keyboard")
        self.mouse = UInput({
            ecodes.EV_KEY: [ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE],
            ecodes.EV_REL: [ecodes.REL_X, ecodes.REL_Y, ecodes.REL_WHEEL],
        }, name="face-slide-control mouse")
        self.tablet = None  # absolute pointer, created on first use

    def key_code(self, name):
        name = name.strip().lower()
        code = self.e.ecodes.get("KEY_" + KEY_ALIASES.get(name, name).upper())
        if code is None:
            raise ValueError(f"Unknown key name {name!r}")
        return code

    def send(self, action):
        codes = [self.key_code(k) for k in action.split("+")]
        for code in codes:
            self.keyboard.write(self.e.EV_KEY, code, 1)
        for code in reversed(codes):
            self.keyboard.write(self.e.EV_KEY, code, 0)
        self.keyboard.syn()

    def move(self, dx, dy):
        self.mouse.write(self.e.EV_REL, self.e.REL_X, dx)
        self.mouse.write(self.e.EV_REL, self.e.REL_Y, dy)
        self.mouse.syn()

    def move_to(self, x, y):
        """Absolute position, 0..1 across the whole desktop (like a VM tablet device)."""
        if self.tablet is None:
            from evdev import AbsInfo
            info = AbsInfo(value=0, min=0, max=ABS_MAX, fuzz=0, flat=0, resolution=0)
            self.tablet = self._UInput({
                self.e.EV_KEY: [self.e.BTN_LEFT, self.e.BTN_RIGHT],
                self.e.EV_ABS: [(self.e.ABS_X, info), (self.e.ABS_Y, info)],
            }, name="face-slide-control tablet")
        self.tablet.write(self.e.EV_ABS, self.e.ABS_X, int(x * ABS_MAX))
        self.tablet.write(self.e.EV_ABS, self.e.ABS_Y, int(y * ABS_MAX))
        self.tablet.syn()

    def scroll(self, notches):
        self.mouse.write(self.e.EV_REL, self.e.REL_WHEEL, notches)
        self.mouse.syn()

    def button(self, name, pressed):
        code = {"left": self.e.BTN_LEFT, "right": self.e.BTN_RIGHT, "middle": self.e.BTN_MIDDLE}[name]
        self.mouse.write(self.e.EV_KEY, code, int(pressed))
        self.mouse.syn()

    def close(self):
        for dev in (self.keyboard, self.mouse, self.tablet):
            if dev is not None:
                dev.close()


class PyAutoGuiBackend:
    """X11 fallback. On Wayland it only reaches XWayland apps, if it connects at all."""

    def __init__(self):
        import pyautogui  # imported lazily: needs a display server
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False
        self.p = pyautogui

    def send(self, action):
        keys = action.split("+")
        if len(keys) > 1:
            self.p.hotkey(*keys)
        else:
            self.p.press(action)

    def move(self, dx, dy):
        self.p.moveRel(dx, dy, _pause=False)

    def move_to(self, x, y):
        w, h = self.p.size()
        self.p.moveTo(int(x * (w - 1)), int(y * (h - 1)), _pause=False)

    def scroll(self, notches):
        self.p.scroll(notches)

    def button(self, name, pressed):
        (self.p.mouseDown if pressed else self.p.mouseUp)(button=name)

    def close(self):
        pass


def create_backend(name):
    if name in ("auto", "uinput"):
        try:
            backend = UInputBackend()
            log.info("Input backend: uinput (virtual keyboard/mouse)")
            return backend
        except Exception as exc:
            if name == "uinput":
                raise
            log.warning("uinput unavailable (%s); falling back to pyautogui", exc)
    backend = PyAutoGuiBackend()
    log.info("Input backend: pyautogui")
    return backend


class InputSender:
    """Sends keys and mouse events. Failures are logged instead of crashing the app."""

    def __init__(self, backend="auto", dry_run=False):
        self.backend_name = backend
        self.dry_run = dry_run
        self._backend = None
        self._unavailable = None
        self._last_error = None

    def prepare(self):
        """Creates the backend up front (a new uinput device needs a moment to register)."""
        self._call("prepare", lambda b: None)

    def send(self, action):
        self._call(f"press {action}", lambda b: b.send(action))

    def move(self, dx, dy):
        self._call(None, lambda b: b.move(dx, dy))

    def move_to(self, x, y):
        self._call(None, lambda b: b.move_to(x, y))

    def scroll(self, notches):
        self._call(f"scroll {'up' if notches > 0 else 'down'} {abs(notches)}", lambda b: b.scroll(notches))

    def button(self, name, pressed):
        self._call(f"{name} button {'down' if pressed else 'up'}", lambda b: b.button(name, pressed))

    def click(self, name):
        self.button(name, True)
        self.button(name, False)

    def close(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def _call(self, description, fn):
        if self.dry_run:
            if description and description != "prepare":
                log.info("[dry-run] would %s", description)
            return
        if self._unavailable:
            return
        if self._backend is None:
            try:
                self._backend = create_backend(self.backend_name)
            except Exception as exc:
                self._unavailable = f"{type(exc).__name__}: {exc}"
                log.error("No input backend available, gestures will only be shown (%s). "
                          "See README 'Troubleshooting'.", self._unavailable)
                return
        try:
            fn(self._backend)
            self._last_error = None
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if message != self._last_error:  # avoid flooding the log every frame
                log.error("Could not send input (%s). See README 'Troubleshooting'.", message)
            self._last_error = message

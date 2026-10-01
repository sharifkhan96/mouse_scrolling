"""System tray icon. Runs as a separate process under the system Python (which has GTK).

Prints the chosen menu command to stdout; reads status text lines from stdin.
Must not import anything from the slidecontrol package.
"""

import sys
import threading

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402

ICON = "camera-web"
ITEMS = [
    ("Pause / resume", "toggle"),
    ("Recalibrate face", "recalibrate"),
    ("Guided setup", "setup"),
    ("Calibrate gaze", "gaze"),
    ("Settings", "settings"),
    (None, None),
    ("On-screen keyboard", "keyboard"),
    ("Presenter overlay", "overlay"),
    ("Laser pointer", "laser"),
    (None, None),
    ("Quit", "quit"),
]


def emit(command):
    print(command, flush=True)


def build_menu():
    menu = Gtk.Menu()
    status = Gtk.MenuItem(label="Starting…")
    status.set_sensitive(False)
    menu.append(status)
    menu.append(Gtk.SeparatorMenuItem())
    for label, command in ITEMS:
        if label is None:
            menu.append(Gtk.SeparatorMenuItem())
            continue
        item = Gtk.MenuItem(label=label)
        item.connect("activate", lambda _w, c=command: emit(c))
        menu.append(item)
    menu.show_all()
    return menu, status


def main():
    menu, status = build_menu()
    holder = {}

    indicator = None
    for namespace in ("AyatanaAppIndicator3", "AppIndicator3"):
        try:
            gi.require_version(namespace, "0.1")
            module = __import__("gi.repository", fromlist=[namespace]).__dict__[namespace]
            indicator = module.Indicator.new("face-slide-control", ICON,
                                             module.IndicatorCategory.APPLICATION_STATUS)
            indicator.set_status(module.IndicatorStatus.ACTIVE)
            indicator.set_menu(menu)
            break
        except (ValueError, ImportError, KeyError, AttributeError):
            continue
    if indicator is None:
        icon = Gtk.StatusIcon.new_from_icon_name(ICON)
        icon.set_tooltip_text("Face Slide Control")
        icon.connect("activate", lambda *_: emit("toggle"))
        icon.connect("popup-menu", lambda i, button, t: menu.popup(None, None, Gtk.StatusIcon.position_menu,
                                                                  i, button, t))
        holder["icon"] = icon

    def set_status(text):
        status.set_label(text)
        if "icon" in holder:
            holder["icon"].set_tooltip_text(f"Face Slide Control – {text}")
        return False

    def read_stdin():
        for line in sys.stdin:
            GLib.idle_add(set_status, line.strip())
        GLib.idle_add(Gtk.main_quit)

    threading.Thread(target=read_stdin, daemon=True).start()
    Gtk.main()


if __name__ == "__main__":
    main()

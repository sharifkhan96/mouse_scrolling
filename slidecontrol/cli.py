"""Command-line entry point."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config, FACE_GESTURES, list_profiles, load_json, load_profile, merge, user_config_path

log = logging.getLogger("slidecontrol")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="face_slide_control.py",
        description="Hands-free control of PDFs, slides and your desktop with face, hand and voice gestures.")
    p.add_argument("--config", help=f"JSON settings file (default: {user_config_path()} if it exists)")
    p.add_argument("--profile", help="settings profile to layer on top, e.g. pdf, slides, video")
    p.add_argument("--list-profiles", action="store_true", help="list available profiles and exit")
    p.add_argument("--dump-config", action="store_true", help="print the effective settings as JSON and exit")
    p.add_argument("--stats", action="store_true", help="print usage statistics of past sessions and exit")
    p.add_argument("--camera", type=int, help="camera index (default 0)")
    p.add_argument("--gestures", help=f"face gestures, comma-separated or 'all' ({', '.join(FACE_GESTURES)})")
    p.add_argument("--dry-run", action="store_true", help="log actions instead of pressing keys")
    p.add_argument("--backend", choices=["auto", "uinput", "pyautogui"], help="how keys and mouse events are sent")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the preview")
    p.add_argument("--no-hand", action="store_true", help="disable hand tracking")
    p.add_argument("--head-scroll", action="store_true", help="scroll by looking up/down")
    p.add_argument("--voice", action="store_true", help="enable offline voice commands")
    p.add_argument("--gaze", action="store_true", help="move the pointer with your eyes (calibrates if needed)")
    p.add_argument("--overlay", action="store_true", help="show the presenter overlay (timer, status)")
    p.add_argument("--no-preview", action="store_true", help="run without the camera preview window")
    p.add_argument("--tray", action="store_true", help="show a system tray icon with a menu")
    p.add_argument("--setup", action="store_true", help="start with the guided setup")
    p.add_argument("--install-autostart", action="store_true",
                   help="start at login (with --tray --no-preview plus any other options given)")
    p.add_argument("--remove-autostart", action="store_true", help="stop starting at login")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def cli_overrides(args):
    data = {}
    if args.camera is not None:
        data["camera"] = args.camera
    if args.gestures:
        data["gestures"] = FACE_GESTURES if args.gestures == "all" else [
            g.strip() for g in args.gestures.split(",") if g.strip()]
    flags = {
        "dry_run": args.dry_run, "head_scroll": args.head_scroll, "voice": args.voice, "gaze": args.gaze,
        "overlay": args.overlay,
    }
    data.update({k: True for k, v in flags.items() if v})
    if args.backend:
        data["input_backend"] = args.backend
    if args.no_mirror:
        data["mirror"] = False
    if args.no_hand:
        data["hand_control"] = False
    if args.no_preview:
        data["show_preview"] = False
    return data


def build_config(args):
    """Returns (cfg, base_data, cli_data, config_path). Layers: defaults < file < profile < CLI."""
    config_path = Path(args.config) if args.config else user_config_path()
    base = load_json(config_path) if (args.config or config_path.is_file()) else {}
    cli = cli_overrides(args)
    profile = args.profile or base.get("profile", "")
    data = merge(base, load_profile(profile)) if profile else dict(base)
    data = merge(data, cli)
    data["profile"] = profile
    return Config.from_dict(data), base, cli, config_path


def autostart_args(argv):
    """The options to start with at login: everything given except the autostart flags."""
    args = [a for a in argv if a not in ("--install-autostart", "--remove-autostart")]
    for flag in ("--tray", "--no-preview"):
        if flag not in args:
            args.append(flag)
    return args


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.stats:
        from .stats import load_sessions, summarize
        print(summarize(load_sessions()))
        return 0
    if args.list_profiles:
        for name in list_profiles():
            print(f"{name:<10} {load_profile(name).get('_description', '')}")
        return 0
    if args.remove_autostart:
        from .system import remove_autostart
        print("Autostart removed." if remove_autostart() else "Autostart was not installed.")
        return 0

    try:
        cfg, base, cli, config_path = build_config(args)
    except (OSError, ValueError) as exc:
        log.error("Invalid configuration: %s", exc)
        return 2

    if args.install_autostart:
        from .system import install_autostart
        print(f"Installed {install_autostart(autostart_args(argv))}")
        return 0
    if args.dump_config:
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0

    log.info("Profile: %s | face gestures: %s", cfg.profile or "default",
             ", ".join(f"{g}->{cfg.actions.get(g)}" for g in cfg.gestures))
    from .app import App
    return App(cfg, base, cli, config_path).run(start_setup=args.setup, tray=args.tray)

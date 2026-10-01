"""Per-session usage statistics, stored as JSON lines."""

import json
import time
from collections import Counter

from .config import data_dir


def stats_path():
    return data_dir() / "sessions.jsonl"


def build_session(start_wall, duration, actions, clicks, near_misses, blinks, profile):
    return {
        "start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_wall)),
        "duration_s": round(duration, 1),
        "profile": profile,
        "actions": dict(actions),
        "clicks": dict(clicks),
        "near_misses": dict(near_misses),
        "blinks": blinks,
    }


def save_session(session, path=None):
    path = path or stats_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(session) + "\n")


def load_sessions(path=None):
    path = path or stats_path()
    if not path.is_file():
        return []
    sessions = []
    with open(path) as fh:
        for line in fh:
            try:
                sessions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return sessions


def summarize(sessions):
    if not sessions:
        return "No sessions recorded yet."
    total = sum(s.get("duration_s", 0) for s in sessions)
    actions, misses, clicks = Counter(), Counter(), Counter()
    blinks = 0
    for s in sessions:
        actions.update(s.get("actions", {}))
        misses.update(s.get("near_misses", {}))
        clicks.update(s.get("clicks", {}))
        blinks += s.get("blinks", 0)
    lines = [
        f"Sessions: {len(sessions)}  (last: {sessions[-1].get('start', '?')})",
        f"Total time: {total / 3600:.1f} h",
        f"Average blink rate: {blinks / (total / 60):.1f} / min" if total >= 60 else "Average blink rate: n/a",
        "",
        f"{'gesture / source':<24}{'fired':>8}{'near misses':>14}{'success':>10}",
    ]
    for name in sorted(set(actions) | set(misses), key=lambda n: -actions[n]):
        fired, missed = actions[name], misses[name]
        rate = f"{fired / (fired + missed):.0%}" if fired + missed else "-"
        lines.append(f"{name:<24}{fired:>8}{missed:>14}{rate:>10}")
    if clicks:
        lines += ["", "Hand clicks: " + ", ".join(f"{k}={v}" for k, v in clicks.most_common())]
    lines += ["", "Near misses: gesture held for more than half the required time, then released."]
    return "\n".join(lines)

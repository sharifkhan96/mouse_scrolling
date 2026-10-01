"""Presence detection (auto-pause), blink-rate monitoring and break reminders."""

from collections import deque

MAX_BLINK = 0.5  # longer closures are deliberate (long blink), not blinks
MIN_BLINK = 0.04
BLINK_WINDOW = 60.0
LOW_BLINK_REPEAT = 300.0  # seconds between low-blink-rate reminders
SECOND_FACE_HOLD = 1.0


class Presence:
    """Decides when control should pause automatically.

    Returns a reason string ("away", "someone else in view") or "" when all is well.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.missing_since = None
        self.crowd_since = None

    def update(self, face_count, now):
        c = self.cfg
        if face_count == 0:
            self.crowd_since = None
            if self.missing_since is None:
                self.missing_since = now
            if c.auto_pause_away and now - self.missing_since >= c.away_seconds:
                return "away"
            return ""
        self.missing_since = None
        if face_count > 1 and c.pause_on_second_face:
            if self.crowd_since is None:
                self.crowd_since = now
            if now - self.crowd_since >= SECOND_FACE_HOLD:
                return "someone else in view"
            return ""
        self.crowd_since = None
        return ""


class BlinkMonitor:
    """Counts natural blinks (both eyes briefly closed)."""

    def __init__(self):
        self.closed_since = None
        self.blinks = deque()
        self.total = 0
        self.start = None

    def update(self, both_closed, now):
        """Returns True when a blink just finished."""
        if self.start is None:
            self.start = now
        while self.blinks and now - self.blinks[0] > BLINK_WINDOW:
            self.blinks.popleft()
        if both_closed:
            if self.closed_since is None:
                self.closed_since = now
            return False
        if self.closed_since is None:
            return False
        duration = now - self.closed_since
        self.closed_since = None
        if MIN_BLINK <= duration <= MAX_BLINK:
            self.blinks.append(now)
            self.total += 1
            return True
        return False

    def rate(self, now):
        """Blinks per minute over the last minute, or None during the first 20 seconds."""
        if self.start is None:
            return None
        elapsed = min(now - self.start, BLINK_WINDOW)
        if elapsed < 20:
            return None
        return len(self.blinks) * 60.0 / elapsed


class Wellness:
    """Break reminders (20-20-20 rule) and low-blink-rate reminders."""

    def __init__(self, cfg, feedback):
        self.cfg = cfg
        self.feedback = feedback
        self.blinks = BlinkMonitor()
        self.present_time = 0.0
        self.last_update = None
        self.last_low_blink_alert = None
        self.reminders = 0

    def update(self, face_present, both_closed, now):
        dt = 0.0 if self.last_update is None else min(now - self.last_update, 1.0)
        self.last_update = now
        if not face_present:
            self.blinks.closed_since = None
            return
        self.blinks.update(both_closed, now)
        self.present_time += dt
        c = self.cfg
        if c.break_reminder_minutes and self.present_time >= c.break_reminder_minutes * 60:
            self.present_time = 0.0
            self.reminders += 1
            self.feedback.play("reminder")
            self.feedback.notify("Time for a short break",
                                 "Look at something about 6 m (20 ft) away for 20 seconds.")
        rate = self.blinks.rate(now)
        if c.low_blink_rate and rate is not None and rate < c.low_blink_rate and \
                now - self.blinks.start >= 120 and (
                    self.last_low_blink_alert is None or now - self.last_low_blink_alert >= LOW_BLINK_REPEAT):
            self.last_low_blink_alert = now
            self.reminders += 1
            self.feedback.notify("Remember to blink", f"Only {rate:.0f} blinks per minute; your eyes may get dry.")

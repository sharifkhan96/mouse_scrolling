"""Hands-free on-screen keyboard logic (row/column scanning)."""

LAYOUT = [
    ["A", "B", "C", "D", "E", "F", "G"],
    ["H", "I", "J", "K", "L", "M", "N"],
    ["O", "P", "Q", "R", "S", "T", "U"],
    ["V", "W", "X", "Y", "Z", ".", ","],
    ["1", "2", "3", "4", "5", "?", "⌫"],
    ["6", "7", "8", "9", "0", "SPACE", "ENTER"],
    ["CLOSE"],
]
SPECIAL = {"⌫": "backspace", "SPACE": "space", "ENTER": "enter", ".": "dot", ",": "comma", "?": "shift+slash"}


def key_action(label):
    """Key name to send for a keyboard label, or "@close"."""
    if label == "CLOSE":
        return "@close"
    return SPECIAL.get(label, label.lower())


class ScanKeyboard:
    """Highlights rows, then keys, in turn. "select" picks the highlighted item;
    "back" returns to row scanning. Works with any single gesture."""

    def __init__(self, interval=1.0, layout=LAYOUT):
        self.layout = layout
        self.interval = interval
        self.row = 0
        self.col = None  # None = scanning rows
        self.last_step = None
        self.typed = ""

    def tick(self, now):
        if self.last_step is None:
            self.last_step = now
        if now - self.last_step < self.interval:
            return
        self.last_step = now
        if self.col is None:
            self.row = (self.row + 1) % len(self.layout)
        else:
            self.col = (self.col + 1) % len(self.layout[self.row])

    def select(self, now):
        """Returns the action to perform, or None when only the row was chosen."""
        self.last_step = now
        if self.col is None:
            if len(self.layout[self.row]) == 1:
                return self._press(self.layout[self.row][0])
            self.col = 0
            return None
        label = self.layout[self.row][self.col]
        self.col = None
        return self._press(label)

    def back(self, now):
        self.last_step = now
        self.col = None

    def press(self, label):
        """Direct selection (e.g. clicking a key with the hand or gaze pointer)."""
        self.col = None
        return self._press(label)

    def _press(self, label):
        action = key_action(label)
        if action == "backspace":
            self.typed = self.typed[:-1]
        elif action == "space":
            self.typed += " "
        elif not action.startswith("@") and action != "enter":
            self.typed += label
        return action

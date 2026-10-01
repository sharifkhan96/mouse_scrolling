"""Offline voice commands with Vosk."""

import json
import logging
import queue
import threading

log = logging.getLogger("slidecontrol")

UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
NUMBER_WORDS = {w: i for i, w in enumerate(UNITS)} | TENS


def words_to_number(words):
    """"one hundred twenty three" -> 123; also accepts digits. None if not a number."""
    if len(words) == 1 and words[0].isdigit():
        return int(words[0])
    total, current = 0, 0
    if not words:
        return None
    for w in words:
        if w in NUMBER_WORDS:
            current += NUMBER_WORDS[w]
        elif w == "hundred":
            current = max(current, 1) * 100
        elif w == "thousand":
            total += max(current, 1) * 1000
            current = 0
        elif w == "and":
            continue
        else:
            return None
    return total + current


def parse_command(text, commands, profiles=()):
    """Maps recognized text to an action, e.g. "go to page twelve" -> "@goto:12"."""
    text = " ".join(text.lower().split())
    if not text:
        return None
    if text in commands:
        return commands[text]
    for prefix in ("go to page ", "go to slide ", "page ", "slide "):
        if text.startswith(prefix):
            n = words_to_number(text[len(prefix):].split())
            return f"@goto:{n}" if n is not None else None
    if text.startswith("profile "):
        name = text[len("profile "):].replace(" ", "_")
        return f"@profile:{name}" if name in profiles else None
    return None


def grammar(commands, profiles=()):
    phrases = set(commands)
    phrases |= {"go to page", "go to slide", "page", "slide", "hundred", "and"} | set(NUMBER_WORDS)
    phrases |= {f"profile {p.replace('_', ' ')}" for p in profiles}
    return sorted(phrases) + ["[unk]"]


class VoiceListener(threading.Thread):
    """Listens to the microphone in the background and queues (text, action) pairs."""

    def __init__(self, model_path, commands, profiles=()):
        super().__init__(daemon=True, name="voice")
        self.model_path = model_path
        self.commands = commands
        self.profiles = list(profiles)
        self.results = queue.Queue()
        self.error = None
        self.listening = False
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            import sounddevice as sd
            import vosk
            vosk.SetLogLevel(-1)
            if not self.model_path.is_dir():
                raise FileNotFoundError(f"speech model not found at {self.model_path} (see README)")
            model = vosk.Model(str(self.model_path))
            rec = vosk.KaldiRecognizer(model, 16000, json.dumps(grammar(self.commands, self.profiles)))
            audio = queue.Queue()

            def callback(data, frames, time_info, status):
                audio.put(bytes(data))

            with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype="int16", channels=1, callback=callback):
                self.listening = True
                log.info("Voice commands: listening")
                while not self._stop_event.is_set():
                    try:
                        data = audio.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if rec.AcceptWaveform(data):
                        text = json.loads(rec.Result()).get("text", "").replace("[unk]", "").strip()
                        if text:
                            action = parse_command(text, self.commands, self.profiles)
                            log.debug("Heard %r -> %s", text, action)
                            self.results.put((text, action))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("Voice commands unavailable: %s", self.error)
        finally:
            self.listening = False

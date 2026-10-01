# Face Slide Control

Hands-free control of PDFs, presentations and your desktop using a normal
webcam and microphone. Face gestures (winks, blinks, eyebrows, head movement),
hand gestures (pointer, pinch-click, swipe, zoom) and offline voice commands are
turned into key presses, scrolling and mouse movement for the app you are using.

Everything runs locally: the camera image and your voice never leave the computer.

```bash
venv1/bin/python face_slide_control.py                 # default: winks scroll, hand = mouse
venv1/bin/python face_slide_control.py --profile pdf   # tuned for reading PDFs
venv1/bin/python face_slide_control.py --setup         # guided setup for your face
```

---

## Contents

1. [Features at a glance](#features-at-a-glance)
2. [Installation](#installation)
3. [Getting started](#getting-started)
4. [Face gestures](#face-gestures)
5. [Hand gestures](#hand-gestures)
6. [Voice commands](#voice-commands)
7. [Extra modes](#extra-modes): head-tilt scrolling, gaze pointer, laser pointer, on-screen keyboard, presenter overlay
8. [Profiles](#profiles)
9. [Settings, guided setup and configuration](#settings-guided-setup-and-configuration)
10. [Presence, eye health and statistics](#presence-eye-health-and-statistics)
11. [Running in the background: tray icon and start at login](#running-in-the-background)
12. [Command-line options and preview keys](#command-line-options-and-preview-keys)
13. [How it works](#how-it-works)
14. [Troubleshooting](#troubleshooting)
15. [Known limitations](#known-limitations)
16. [Development and tests](#development-and-tests)

---

## Features at a glance

| Area | What you can do |
|---|---|
| **Face** | Wink to scroll or change slides (normal blinks are ignored), long blink to pause, raise eyebrows, smile, open mouth, turn or tilt your head |
| **Hands** | Move the mouse pointer, pinch to click and drag, right-click, swipe to change pages, two-finger scroll, two-hand zoom, open palm to pause, fist to resume |
| **Voice** | "next", "back", "go to page twelve", "zoom in", "pause", "profile slides" and more, offline |
| **Head-tilt scrolling** | Look down to scroll down, up to scroll up; the further you tilt, the faster it goes |
| **Gaze pointer** | After a 9-dot calibration your eyes move the mouse pointer, with optional dwell-to-click |
| **Laser pointer** | A red dot follows your index finger during presentations |
| **On-screen keyboard** | Type with winks only (row/column scanning) or click keys with the hand/gaze pointer |
| **Presenter overlay** | Small always-on-top panel: timer, status, profile, slide counter, last gesture |
| **Profiles** | Ready-made settings for PDFs, slides and videos; switch with a key or by voice |
| **Guided setup** | Records your own winks, brows, smile and mouth and tunes the thresholds for you |
| **Settings window** | Sliders and switches for everything, applied live and saved to a file |
| **Presence & health** | Auto-pause when you look away or someone else appears; blink-rate and 20-20-20 break reminders |
| **Feedback & stats** | Sounds and desktop notifications; per-session statistics including near-miss gestures |
| **Desktop integration** | Tray icon menu, run without a preview window, start automatically at login |

## Installation

Requirements: Linux (tested on Ubuntu with GNOME on Wayland), Python 3.12, a webcam,
optionally a microphone.

```bash
git clone https://github.com/sharifkhan96/mouse_scrolling.git
cd mouse_scrolling
python3 -m venv venv1
venv1/bin/pip install -r requirements.txt
```

**Voice commands** also need the small English speech model (≈ 40 MB download, 68 MB unpacked):

```bash
mkdir -p models && cd models
curl -LO https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip && rm vosk-model-small-en-us-0.15.zip
```

**Sending keys and mouse events** uses the Linux virtual input device `/dev/uinput`
(through the `evdev` package). It works on both Wayland and X11 and in every app. On
most desktops your user already has access; if not, see [Troubleshooting](#troubleshooting).

**Optional system packages**: `python3-gi` (tray icon, usually preinstalled),
`gir1.2-ayatanaappindicator3-0.1` (nicer tray icon on Ubuntu), `pipewire` / `pw-play`
(sounds) and `libnotify-bin` / `notify-send` (notifications).

## Getting started

1. Start the program: `venv1/bin/python face_slide_control.py`.
2. A preview window shows the camera. For about 3 seconds, look at the screen with
   your eyes open and a relaxed face while it says **CALIBRATING**.
3. When it says **ACTIVE**, click the window you want to control, such as your PDF or
   presentation, so that it has keyboard focus. The preview window takes focus when it opens.
4. Wink your **left eye** to scroll down and your **right eye** to scroll up. Hold the
   wink to keep scrolling.
5. Close both eyes for about 1.2 s to pause or resume.
6. To quit, click the preview window and press `q`.

For the best results, run the guided setup once (`--setup`, or press `g`). It measures
your own gestures and saves thresholds that fit your face.

Not sure it will work? Use `--dry-run`: gestures are shown and logged, but no keys are pressed.

## Face gestures

"Left" and "right" always mean **your own** left and right.

| Gesture | How | Default action | On by default |
|---|---|---|---|
| `wink_left` | close left eye ~0.2 s, keep right eye open | `down` arrow, repeats while held | ✔ |
| `wink_right` | close right eye ~0.2 s | `up` arrow, repeats while held | ✔ |
| `long_blink` | close both eyes ~1.2 s | pause / resume | ✔ |
| `brow_raise` | raise your eyebrows ~0.4 s | `pagedown` | profiles |
| `smile` | smile widely with your mouth closed ~0.6 s | `shift+print` (screenshot) | |
| `mouth_open` | open your mouth wide ~0.6 s | `b` (blank screen in slide shows) | profiles |
| `head_left` / `head_right` | turn your head ~0.35 s | `up` / `down` | |
| `tilt_left` / `tilt_right` | tilt your head toward a shoulder | `up` / `down` | |

Enable gestures with `--gestures wink_left,wink_right,brow_raise` or `--gestures all`, in
the settings window (*Gestures* tab), or in a config file / profile.

How it avoids accidents:

* **Normal blinks are ignored.** A wink needs one eye closed while the other stays open.
* **Hold times.** Every gesture must be held briefly, so a twitch does not count.
* **Fire once, then release.** A gesture fires once and must be released before it can
  fire again. Scrolling actions (arrow keys, page keys, wheel, zoom) instead auto-repeat
  while held, like holding a key.
* **Cooldown.** At least 0.6 s must pass between two *different* actions.
* **Relative measurements.** Eye openness is the Eye Aspect Ratio compared with *your*
  calibrated open eye, so it does not depend on how far you sit from the camera.

## Hand gestures

Hand control is on by default; disable with `--no-hand` or the `h` key.

| Hand | Result |
|---|---|
| Move your hand (pointing, relaxed) | moves the mouse pointer like an air trackpad |
| Pinch thumb + index finger | left button down; release the pinch to release it (click, or hold to drag) |
| Pinch thumb + middle finger | right click |
| **Fist** | clutch: move your hand without moving the pointer, to reposition it |
| **Two fingers up** (index + middle) and move up/down | scroll |
| **Open palm**, quick sideways swipe | `swipe_left` → `pagedown`, `swipe_right` → `pageup` |
| **Open palm** held still ~1 s | pause (`palm_hold`) |
| **Fist** held ~1 s while paused | resume (`fist_hold`) |
| **Both hands pinching**, pull apart / push together | zoom in / out (`ctrl+=` / `ctrl+-`) |

The pointer follows your **middle knuckle**, which barely moves when you pinch, so
clicks land where you aimed. If your hand leaves the camera view, or control pauses,
a held mouse button is always released.

## Voice commands

Start with `--voice`, press `v`, or switch it on in the settings. Recognition runs
offline with [Vosk](https://alphacephei.com/vosk/) and only listens for the phrases
below, which makes it fast and accurate.

| Say | Does |
|---|---|
| "next", "next page", "next slide" | `pagedown` |
| "back", "previous", "previous page/slide" | `pageup` |
| "down", "up" | arrow keys |
| "scroll down", "scroll up" | mouse wheel |
| "first page", "last page" | `home`, `end` |
| "go to page twelve", "page one hundred five" | types the page number + Enter (in Evince, the `pdf` profile opens the page box first with `ctrl+l`) |
| "zoom in", "zoom out", "full screen" | `ctrl+=`, `ctrl+-`, `f11` |
| "screenshot" | `shift+print` |
| "pause" / "stop", "resume" / "start" | pause and resume control |
| "laser", "keyboard", "overlay" | toggle those modes |
| "calibrate", "setup", "settings" | recalibrate, guided setup, settings window |
| "profile pdf", "profile slides", "profile video" | switch profile |

The last heard phrase is shown in the preview. Add or change phrases under
`voice_commands` in a config file.

## Extra modes

### Head-tilt scrolling
`--head-scroll`, the `t` key, or the `pdf` profile. Look slightly **down** to scroll down
and **up** to scroll up. There is a dead zone around your normal head position, and the
speed grows the further you tilt. It uses the mouse wheel, so it scrolls the window
under the pointer.

### Gaze pointer
`--gaze`, the `e` key, or *Calibrate gaze* in the settings. A full-screen calibration
shows 9 dots: keep your head still and look at each dot until it turns red. Your
eyes then move the mouse pointer, and the calibration is saved for next time. Turn on
*Gaze dwell click* to click by looking at one spot for 1.2 s. While gaze is active, hand
movement no longer moves the pointer, but pinch clicks still work. With an ordinary webcam,
accuracy is rough, so it suits large targets better than small buttons.

### Laser pointer
Press `l` or say "laser". A red dot follows your index fingertip on the screen during
a presentation. The real mouse pointer does not move and nothing is clicked.

### On-screen keyboard
Press `k` or say "keyboard". A keyboard appears at the bottom of the screen and
highlights one row at a time: **wink left** selects the row, then the keys in it are
highlighted one by one, and **wink left** again types that key. **Wink right** goes back
to row scanning. You can also click keys with the hand or gaze pointer. Typing goes to
the window that has keyboard focus.

### Presenter overlay
`--overlay`, the `o` key, "overlay", or the `slides` profile. A small, always-on-top,
draggable panel shows the elapsed time, status, profile, a slide counter
(next minus previous) and the last gesture.

## Profiles

Profiles are sets of settings for a type of task, layered on top of your normal settings.

| Profile | For | Highlights |
|---|---|---|
| `pdf` | Evince and other PDF readers | winks = arrow keys with fast repeat, head-tilt scrolling, eyebrows = next page, "go to page" support |
| `slides` | Impress, PowerPoint, Google Slides | winks = one slide (`pagedown`/`pageup`) without repeat, open mouth = blank screen, presenter overlay |
| `video` | YouTube, VLC, mpv | left wink = play/pause, head turn = seek back/forward, smile = fullscreen; no auto-pause when you look away |

Use `--profile pdf`, press `1`–`9` in the preview (alphabetical order: 1 = pdf, 2 = slides,
3 = video), or say "profile slides". List them with `--list-profiles`.

To make your own, put a JSON file in `~/.config/face-slide-control/profiles/` (or in
`profiles/`). It only needs the settings that differ, for example:

```json
{
  "_description": "Comics: big steps",
  "gestures": ["wink_left", "wink_right", "long_blink"],
  "actions": {"wink_left": "pagedown", "wink_right": "pageup"},
  "repeat_enabled": false
}
```

## Settings, guided setup and configuration

**Settings window** (`s` key, tray menu or "settings"): sliders for thresholds, hold times,
cooldown, repeat speed, pointer speed, pinch sensitivity and more; switches for every
feature; and a *Gestures* tab to choose which face gestures are active. Changes apply
immediately. **Save** writes them to `~/.config/face-slide-control/config.json`, which is
loaded automatically next time.

**Guided setup** (`--setup`, `g` key, tray menu or "setup") takes about 20 seconds. It
asks you to relax, wink left, wink right, raise your eyebrows, smile and open your
mouth, then calculates thresholds that fit your face. They are applied immediately
and saved.

**Settings are layered** in this order; later layers win:

1. built-in defaults
2. your config file: `~/.config/face-slide-control/config.json`, or `--config file.json`
3. the profile (`--profile` or `"profile"` in the config file)
4. command-line options

`--dump-config` prints the effective settings. Useful keys:

| Key | Meaning (default) |
|---|---|
| `close_ratio` / `open_ratio` | eye counts as closed below / open above this fraction of your open eye (0.55 / 0.65) |
| `wink_hold`, `long_blink_hold` | seconds to hold (0.2, 1.2) |
| `cooldown` | seconds between different actions (0.6) |
| `repeat_enabled`, `repeat_delay`, `repeat_interval` | auto-repeat while held (on, 0.5 s, 0.12 s) |
| `gestures` | active face gestures |
| `actions` | gesture → action map, see below |
| `voice_commands` | phrase → action map |
| `pointer_speed`, `pinch_threshold`, `swipe_distance` | hand control tuning |
| `head_scroll_deadzone`, `head_scroll_speed` | head-tilt scrolling tuning |
| `auto_pause_away`, `away_seconds`, `pause_on_second_face` | presence |
| `break_reminder_minutes`, `low_blink_rate` | eye-health reminders (0 = off) |
| `sounds`, `notifications` | feedback |
| `goto_prefix` | keys pressed before typing a page number ("ctrl+l" for Evince) |

**Actions** can be:

* key names: `down`, `pagedown`, `space`, `b`, `f5`, `home` …
* key combinations: `ctrl+equal`, `shift+print`, `ctrl+shift+f5`
* internal actions: `@scroll_down`, `@scroll_up` (mouse wheel), `@pause`, `@pause_on`,
  `@resume`, `@recalibrate`, `@laser`, `@keyboard`, `@overlay`, `@settings`, `@setup`,
  `@gaze_calibrate`, `@profile:<name>`, `@goto:<page>`

## Presence, eye health and statistics

* **Auto-pause when you look away**: if no face is seen for 3 s, control pauses, and
  it resumes when you come back. You can also pause when a second person appears
  (`pause_on_second_face`). With several people in view, the closest face is used.
* **Blink rate**: natural blinks are counted and shown in the presenter overlay. If you
  blink less than 8 times per minute (common when staring at screens), you get a reminder.
* **Break reminder**: every 20 minutes of screen time, a notification suggests looking
  20 feet (6 m) away for 20 seconds.
* **Sounds**: short sounds confirm actions, pause and resume, and calibration.
* **Statistics**: every session is saved to `~/.local/share/face-slide-control/sessions.jsonl`.
  `--stats` shows totals per gesture, including **near misses** (a gesture held for more
  than half the required time and then released). A low success rate for a gesture
  means its threshold or hold time should be adjusted.

## Running in the background

* `--tray` adds a tray icon with a menu: pause/resume, recalibrate, guided setup, gaze
  calibration, settings, keyboard, overlay, laser, quit.
* `--no-preview` runs without the camera window. Control it by gestures, voice or the tray menu.
* `--install-autostart` starts the program at login, with `--tray --no-preview` plus any
  other options you give, e.g. `--install-autostart --profile pdf --voice`.
  `--remove-autostart` undoes it.

## Command-line options and preview keys

```
--profile NAME        --list-profiles       --config FILE        --dump-config
--setup               --stats               --dry-run            --camera N
--gestures LIST|all   --no-hand             --head-scroll        --voice
--gaze                --overlay             --tray               --no-preview
--no-mirror           --backend auto|uinput|pyautogui
--install-autostart   --remove-autostart    -v / --verbose
```

Keys in the preview window:

| Key | | Key | |
|---|---|---|---|
| `q` / Esc | quit | `k` | on-screen keyboard |
| `p` | pause / resume | `l` | laser pointer |
| `c` | recalibrate face | `o` | presenter overlay |
| `g` | guided setup | `h` | hand control on/off |
| `s` | settings window | `t` | head-tilt scrolling on/off |
| `e` | gaze calibration | `v` | voice commands on/off |
| `d` | dry-run on/off | `m` | face points on/off |
| `1`–`9` | switch profile | | |

The preview shows the status, active modes, eye openness bars with threshold marks,
how far each gesture has progressed, the hand pose, the last voice phrase and the last action.

## How it works

```
webcam ──► MediaPipe FaceMesh (478 points) ──► face features ──► gesture engine ──┐
       └─► MediaPipe Hands (21 points/hand) ──► hand controller ─────────────────┤
microphone ──► Vosk (offline, restricted grammar) ──► voice commands ────────────┤
                                                                                 ▼
                            dispatcher (pause, cooldown, repeat) ──► virtual keyboard / mouse
                                                                     (/dev/uinput)
```

* **Face features** (`slidecontrol/geometry.py`): eye aspect ratio per eye, mouth
  opening, head yaw/pitch/roll from nose and cheek positions, eyebrow height, smile
  width, and iris position for gaze. All are normalised by face size.
* **Calibration**: the median of ~3 s of a neutral face is your baseline; most gestures
  are measured relative to it.
* **Gesture engine** (`gestures.py`): each gesture has a hold timer that fires once,
  or repeats for scrolling actions, and is reset when the face is lost.
* **Hand controller** (`hands.py`): recognises poses (point, pinch, two fingers, palm,
  fist) from which fingers are extended, then drives the pointer, clicks, scrolling,
  swipes and zoom.
* **Gaze** (`motion.py`): a small regression model maps iris position and head pose to
  screen coordinates, trained from the 9-dot calibration.
* **Input** (`inputs.py`): creates a virtual keyboard, mouse and absolute-pointer tablet
  through `/dev/uinput`, so it works on Wayland. pyautogui is a fallback for X11.

Project layout:

```
face_slide_control.py    launcher (python face_slide_control.py ...)
slidecontrol/
  cli.py                 command-line options, config layering
  app.py                 camera loop; ties all features together; preview HUD
  config.py              settings, defaults, profiles, file locations
  geometry.py            face measurements from landmarks
  gestures.py            calibration, hold triggers, gesture engine, dispatcher
  hands.py               hand poses, pointer, clicks, swipe, scroll, zoom, laser
  motion.py              head-tilt scrolling, gaze model and calibration
  voice.py               Vosk listener, phrase/number parsing
  setup_wizard.py        guided setup and threshold derivation
  keyboard.py            scanning keyboard logic
  wellness.py            presence (auto-pause), blink counting, reminders
  stats.py               session statistics
  feedback.py            sounds and notifications
  inputs.py              uinput / pyautogui backends
  ui.py                  Tk windows: overlay, laser dot, settings, setup, gaze, keyboard
  system.py              tray icon process, autostart
  tray_helper.py         tray icon (runs under the system Python with GTK)
profiles/                pdf.json, slides.json, video.json
tests/                   unit tests
models/                  speech model (downloaded, not in git)
```

`mouse_scrolling.py` and `bidirectional-eye-control.py` are the original prototypes,
kept for reference.

### Early prototype tests (March 2025)

The first version (`mouse_scrolling.py`) moved to the next line with a wink:

![Screenshot from 2025-03-12 09-29-22](https://github.com/user-attachments/assets/aa0421e1-52f2-4721-9b10-d18b72c6bef9)

Same test, with the slides moving:

![Screenshot from 2025-03-12 09-31-29](https://github.com/user-attachments/assets/4d8fef67-2714-4df0-84f0-b30c23342fa2)

## Troubleshooting

**Nothing happens when I wink.**
Look at the bottom of the preview after a wink.
* If it shows `wink_left -> down`, the gesture worked, but the target window does not
  have keyboard focus. Click on it.
* If nothing shows, watch the *left eye / right eye* bars while you wink. The closed eye
  must drop below the first red mark, and the other eye must stay above the second.
  Run the guided setup (`g`), or press `c` to recalibrate while looking straight ahead.

**The PDF scrolls only a little.** Hold the wink: after about 0.7 s it starts repeating. Or use
`--profile pdf`, which repeats faster and adds head-tilt scrolling.

**"Could not send input" / "No input backend available".**
Check that `evdev` is installed in the venv and that you can write to `/dev/uinput`
(`ls -l /dev/uinput`; your user needs an ACL entry or the `input` group). Without it,
pyautogui is used, which on Wayland only reaches some apps, if it connects at all.

**"Could not open camera".** Another program, often another copy of this one, is
using the webcam. Close it, or choose another camera with `--camera 1`.

**Voice does nothing.** Check that the model folder exists in `models/` and the
microphone works (`arecord -d 3 test.wav`). Speak a listed phrase clearly. The preview
shows what was heard.

**Tray icon does not show.** On GNOME you need the *AppIndicator* extension (enabled by
default on Ubuntu). Installing `gir1.2-ayatanaappindicator3-0.1` gives the most reliable icon.

**Gaze pointer is jumpy or offset.** Recalibrate (`e`) with your head in its usual
position and good, even light on your face. Increase `gaze_smoothing` in the settings.

## Known limitations

* **No automatic profile switching by app.** GNOME on Wayland does not let programs see
  which window is focused, so profiles are switched by key, voice, command line or config.
* **Gaze tracking is approximate** with a normal webcam. Use it for large targets.
* **Tray icon:** without the AppIndicator library, a legacy GTK status icon is used.
  Some desktops do not show it.
* **On-screen windows** (overlay, keyboard, laser dot, settings) use Tk through XWayland.
  The laser dot and keyboard are positioned windows, so they cannot be made click-through.
* Lighting matters: a lamp behind you or a very dark room makes detection less reliable.

## Development and tests

```bash
venv1/bin/python -m unittest discover -s tests
```

The tests cover the gesture logic, dispatching, hand poses and gestures, head scroll,
gaze calibration, voice parsing, guided setup, scanning keyboard, presence and
wellness, stats, config layering, profiles and autostart. They don't need a camera
or microphone.

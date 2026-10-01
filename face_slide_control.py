"""Launcher: hands-free control with face, hand and voice gestures.

Run `python face_slide_control.py --help` for options. The code lives in the
slidecontrol/ package; see README.md for a full description.
"""

import sys

from slidecontrol.cli import main

if __name__ == "__main__":
    sys.exit(main())

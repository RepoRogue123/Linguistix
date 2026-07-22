"""Put the package root on sys.path so training scripts can import ml_website.

Imported for its side effect only. Lets ``python training/train_encoder.py`` work
from any working directory without requiring an editable install.
"""

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

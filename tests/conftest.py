"""Make ``autocurate`` (and the optional sibling ``judgecurate``) importable
without installing, so the suite runs from a fresh checkout."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

_SIB = os.path.join(ROOT, "..", "LLM_as_Judge_Pretraining_Data_Curation", "src")
if os.path.isdir(_SIB):
    sys.path.insert(0, os.path.abspath(_SIB))

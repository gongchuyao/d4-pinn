"""
Environment bootstrap. Import this BEFORE numpy/torch in every entry-point
script. It silently sets the two OS env-vars that suppress the
`OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
already initialized` error on Windows + Anaconda environments.

Usage in scripts:
    import _bootstrap   # noqa: F401
    import numpy, torch, ...
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Avoid noisy MKL warnings on some Windows installations.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

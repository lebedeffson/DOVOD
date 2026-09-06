"""Opt-in RNG pinning for the post-freeze AMLGym reproducibility amendment.

Python imports ``sitecustomize`` automatically at interpreter startup when this
package directory is on ``sys.path``.  Nothing changes unless
``DOVOD_CONFIRMATORY_SEED`` is set by the confirmatory workflow.  In that case
we seed Python and NumPy before AMLGym is imported; ROSAME jobs additionally
seed PyTorch.
"""

from __future__ import annotations

import os
import random

_seed_text = os.environ.get("DOVOD_CONFIRMATORY_SEED")
if _seed_text is not None:
    _seed = int(_seed_text)
    random.seed(_seed)

    try:
        import numpy as _np
    except ImportError:
        _np = None
    if _np is not None:
        _np.random.seed(_seed % (2**32 - 1))

    if os.environ.get("DOVOD_CONFIRMATORY_TORCH") == "1":
        try:
            import torch as _torch
        except ImportError:
            _torch = None
        if _torch is not None:
            _torch.manual_seed(_seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(_seed)

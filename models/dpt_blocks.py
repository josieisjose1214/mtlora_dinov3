import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_single_task_root = os.path.join(_repo_root, "single_task")
if _single_task_root not in sys.path:
    sys.path.insert(0, _single_task_root)

from dpt_blocks import FeatureFusionBlock_custom, Slice, Transpose  # noqa: F401

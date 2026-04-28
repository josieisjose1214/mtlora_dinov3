# PET 原仓库使用 util.misc，此处转发到 pet_util.misc
import sys
import os
# 确保项目根在 path 中以便 pet_util 可被找到
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from pet_util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    get_world_size,
    is_dist_avail_and_initialized,
)

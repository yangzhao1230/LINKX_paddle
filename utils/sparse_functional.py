import warnings
import paddle
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers.tensor import fill_constant

from paddle.static import Variable

from paddle import _C_ops, _legacy_C_ops
from paddle.framework import in_dynamic_mode
from paddle.tensor.creation import full
from paddle.framework import core
from paddle.fluid.framework import _in_legacy_dygraph
from paddle.static import default_main_program

def sparse_linear(x, weight, bias=None, name=None):
    
    return paddle.sparse.matmul(x, weight) + bias
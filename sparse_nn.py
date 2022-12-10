import paddle
from paddle.nn import Layer
from utils.sparse_functional import sparse_linear

class Sparse_Linear(Layer):
    
    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        bias_attr=None,
        name=None,
    ):
        super(Sparse_Linear, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        self.bias = self.create_parameter(
            shape=[out_features],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )
        self.name = name

    def forward(self, input):
        out = sparse_linear(
            x=input, weight=self.weight, bias=self.bias, name=self.name
        )
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'in_features={}, out_features={}, dtype={}{}'.format(
            self.weight.shape[0], self.weight.shape[1], self._dtype, name_str
        )
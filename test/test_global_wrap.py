import oneflow
import sys
sys.path.append(r'../one-fx')
import fx
import builtins

import modules
from modules import MyModule
from utils import add_n

with onefx.global_wrap(len, builtins):
    without_activation = MyModule(do_activation=False)
    with_activation = MyModule(do_activation=True)

    traced_without_activation = onefx.symbolic_trace(without_activation)
    print(traced_without_activation.code)
    """
    wrap("len")
    wrap("oneflow._oneflow_internal._C.relu")

    def forward(self, x):
        linear = self.linear(x);  x = None
        getattr_1 = linear.shape
        len_1 = len(getattr_1);  getattr_1 = None
        relu = oneflow._oneflow_internal._C.relu(linear)
        add = relu + len_1;  relu = len_1 = None
        getattr_2 = linear.shape;  linear = None
        len_2 = len(getattr_2);  getattr_2 = None
        add_1 = add + len_2;  add = len_2 = None
        return add_1
    """

    traced_with_activation = onefx.symbolic_trace(with_activation)
    print(traced_with_activation.code)
    """
    wrap("len")
    wrap("oneflow._oneflow_internal._C.relu")

    def forward(self, x):
        linear = self.linear(x);  x = None
        getattr_1 = linear.shape
        len_1 = len(getattr_1);  getattr_1 = None
        relu = oneflow._oneflow_internal._C.relu(linear);  linear = None
        relu_1 = oneflow._oneflow_internal._C.relu(relu)
        add = relu_1 + len_1;  relu_1 = len_1 = None
        getattr_2 = relu.shape;  relu = None
        len_2 = len(getattr_2);  getattr_2 = None
        add_1 = add + len_2;  add = len_2 = None
        return add_1
    """
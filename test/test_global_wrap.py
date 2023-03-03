import oneflow
import sys
sys.path.append(r'../one-fx')
import onefx
import builtins

from modules import MyModule

with onefx.global_wrap(len):
    without_activation = MyModule(do_activation=False)
    with_activation = MyModule(do_activation=True)

    traced_without_activation = onefx.symbolic_trace(without_activation)
    print(traced_without_activation.code)
    """
    wrap("len")

    def forward(self, x):
        linear = self.linear(x);  x = None
        getattr_1 = linear.shape
        len_1 = len(getattr_1);  getattr_1 = None
        add = linear + len_1;  linear = len_1 = None
        return add
    """

    traced_with_activation = onefx.symbolic_trace(with_activation)
    print(traced_with_activation.code)
    """
    wrap("len")
    wrap("oneflow.relu")

    def forward(self, x):
        linear = self.linear(x);  x = None
        getattr_1 = linear.shape
        len_1 = len(getattr_1);  getattr_1 = None
        relu = oneflow.relu(linear);  linear = None
        add = relu + len_1;  relu = len_1 = None
        return add
    """
    
    len_id = id(len)
    
assert len_id != id(len)
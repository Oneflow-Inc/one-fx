import oneflow
import sys
sys.path.append(r'../one-fx')
import onefx

def wrap_test_func(x):
    return x

class MyModule(oneflow.nn.Module):
    def __init__(self, do_activation : bool = False):
        super().__init__()
        self.do_activation = do_activation
        self.linear = oneflow.nn.Linear(512, 512)

    def forward(self, x):
        x = self.linear(x)
        y = oneflow.ones([2, 3])

        if self.do_activation:
            x = oneflow.relu(x)
        return y

without_activation = MyModule(do_activation=False)
with_activation = MyModule(do_activation=True)

traced_without_activation = onefx.symbolic_trace(without_activation)
print(traced_without_activation.code)
"""
def forward(self, x):
    linear = self.linear(x);  x = None
    _tensor_constant0 = self._tensor_constant0
    return _tensor_constant0
"""

traced_with_activation = onefx.symbolic_trace(with_activation)
print(traced_with_activation.code)
"""
wrap("oneflow.relu")

def forward(self, x):
    linear = self.linear(x);  x = None
    relu = oneflow.relu(linear);  linear = None
    _tensor_constant0 = self._tensor_constant0
    return _tensor_constant0
"""
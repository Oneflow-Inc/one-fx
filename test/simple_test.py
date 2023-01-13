import oneflow
import sys
sys.path.append(r'../one-fx')
import fx

def wrap_test_func(x):
    return x

class MyModule(oneflow.nn.Module):
    def __init__(self, do_activation : bool = False):
        super().__init__()
        self.do_activation = do_activation
        self.linear = oneflow.nn.Linear(512, 512)

    def forward(self, x):
        x = self.linear(x)
        x = oneflow.relu(x)
        y = oneflow.ones([2, 3])

        if self.do_activation:
            x = oneflow.relu(x)
        return y

without_activation = MyModule(do_activation=False)
with_activation = MyModule(do_activation=True)

traced_without_activation = fx.symbolic_trace(without_activation)
print(traced_without_activation.code)
"""
def forward(self, x):
    linear = self.linear(x);  x = None
    return linear
"""

traced_with_activation = fx.symbolic_trace(with_activation)
print(traced_with_activation.code)
"""
wrap("oneflow._oneflow_internal._C.relu")

def forward(self, x):
    linear = self.linear(x);  x = None
    relu = oneflow._oneflow_internal._C.relu(linear);  linear = None
    return relu
"""
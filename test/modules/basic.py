import oneflow
import sys
sys.path.append(r'../one-fx')
from test.utils.basic import add_n

class MyModule(oneflow.nn.Module):
    def __init__(self, do_activation : bool = False):
        super().__init__()
        self.do_activation = do_activation
        self.linear = oneflow.nn.Linear(512, 512)

    def forward(self, x):
        x = self.linear(x)
        y = len(x.shape)

        if self.do_activation:
            x = oneflow.relu(x)
        return add_n(x, y)
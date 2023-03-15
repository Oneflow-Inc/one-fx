# one-fx

[![PyPI version](https://img.shields.io/pypi/v/onefx.svg)](https://pypi.org/project/onefx/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/onefx.svg)](https://pypi.org/project/onefx/)
[![PyPI license](https://img.shields.io/pypi/l/onefx.svg)](https://pypi.org/project/onefx/)

A toolkit for developers to simplify the transformation of nn.Module instances. It is modified from `Pytorch.fx`.


## install

```shell
pip install onefx
```

[Oneflow](https://github.com/Oneflow-Inc/oneflow) has now add `one-fx` as default dependency. You can also install oneflow and use it as `oneflow.fx`.

## usage

The following code shows the basic usage. For more examples, please refer to `https://github.com/Oneflow-Inc/one-fx/tree/main/onefx/exmaples`.

```python
import oneflow
import onefx as fx

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

traced_without_activation = onefx.symbolic_trace(without_activation)
print(traced_without_activation.code)
"""
def forward(self, x):
    linear = self.linear(x);  x = None
    return linear
"""

traced_with_activation = onefx.symbolic_trace(with_activation)
print(traced_with_activation.code)
"""
wrap("oneflow._oneflow_internal._C.relu")

def forward(self, x):
    linear = self.linear(x);  x = None
    relu = oneflow._oneflow_internal._C.relu(linear);  linear = None
    return relu
"""
```

## version map

| oneflow | one-fx |
| ------- | ------- |
| >=0.9.0 | 0.0.2, 0.0.3 |
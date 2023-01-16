"""
Modified from https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/fake_tensor_prop.py
"""

from typing import Optional

import onefx
from onefx import Node
from onefx._compatibility import compatibility
from oneflow._subclasses.fake_tensor import FakeTensorMode

# Delete

__all__ = ['FakeTensorProp']

@compatibility(is_backward_compatible=False)
class FakeTensorProp(onefx.Interpreter):
    """
    Execute an FX graph Node-by-Node and record a fake tensor representing
    the metadata for the node.  Unlike ShapeProp, (1) this propagation
    is cheap--it does the propagation with meta tensors which do not actually
    store data, and (2) the fake tensors have much more fine grained information,
    e.g., they have accurate alias information that can be consulted by looking
    at the storages.

    Args:
         module (GraphModule): The module to be executed
         mode (Optional[FakeTensorMode]): The dispatch mode used to execute computation indicated by each FX Node.
    """
    def __init__(self, module: onefx.GraphModule, mode: Optional[FakeTensorMode] = None):
        super().__init__(module)
        if mode is None:
            mode = FakeTensorMode()
        self._mode = mode

    def run_node(self, n: Node):
        result = super().run_node(n)
        n.meta['val'] = result
        return result

    def propagate(self, *args):
        with self._mode:
            fake_args = [self._mode.from_tensor(a) for a in args]
            return super().run(*fake_args)

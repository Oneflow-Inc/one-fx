'''
Modified from https://github.com/pytorch/examples/blob/main/fx/model_tracer.py
'''

"""fx
Recording Module Hierarchy With a Custom Tracer

In this example, we are going to define a custom `fx.Tracer` instance that--
for each recorded operation--also notes down the qualified name of the module
from which that operation originated. The _qualified name_ is the path to the
Module from the root module.
"""
import oneflow
import sys
sys.path.append(r'../one-fx')
import onefx
from typing import Any, Callable, Dict, Optional, Tuple

class ModulePathTracer(onefx.Tracer):
    """
    ModulePathTracer is an FX tracer that--for each operation--also records
    the qualified name of the Module from which the operation originated.
    """

    # The current qualified name of the Module being traced. The top-level
    # module is signified by empty string. This is updated when entering
    # call_module and restored when exiting call_module
    current_module_qualified_name : str = ''
    # A map from FX Node to the qualname of the Module from which it
    # originated. This is recorded by `create_proxy` when recording an
    # operation
    node_to_originating_module : Dict[onefx.Node, str] = {}

    def call_module(self, m: oneflow.nn.Module, forward: Callable[..., Any],
                    args : Tuple[Any, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Override of Tracer.call_module (see
        https://pyoneflow.org/docs/stable/fx.html#onefx.Tracer.call_module).

        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Installs the qualified name of the caller in `current_module_qualified_name`
           for retrieval by `create_proxy`
        3) Delegates into the normal Tracer.call_module method
        4) Restores the caller's qualified name into current_module_qualified_name
        """
        old_qualname = self.current_module_qualified_name
        try:
            self.current_module_qualified_name = self.path_of_module(m)
            return super().call_module(m, forward, args, kwargs)
        finally:
            self.current_module_qualified_name = old_qualname

    def create_proxy(self, kind: str, target: onefx.node.Target, args: Tuple[Any, ...],
                     kwargs: Dict[str, Any], name: Optional[str] = None, type_expr: Optional[Any] = None):
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_originating_module`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_originating_module[proxy.node] = self.current_module_qualified_name
        return proxy


# Testing: let's see how this works on a oneflowvision ResNet18 model
import flowvision.models as models

# Model under test
rn18 = models.resnet18()

# Instantiate our ModulePathTracer and use that to trace our ResNet18
tracer = ModulePathTracer()
traced_rn18 = tracer.trace(rn18)

# Print (node, module qualified name) for every node in the Graph
for node in traced_rn18.nodes:
    module_qualname = tracer.node_to_originating_module.get(node)
    print('Node', node, 'is from module', module_qualname)
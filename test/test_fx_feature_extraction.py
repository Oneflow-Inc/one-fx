import math
import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Mapping,Sequence
import random
import pytest
import builtins

import oneflow
import flowvision
import oneflow.nn as nn
import sys
sys.path.append(r'../one-fx')
import onefx
from onefx.graph_module import _copy_attr


__all__ = ["create_feature_extractor", "get_graph_node_names"]


class LeafModuleAwareTracer(onefx.Tracer):
    """
    An onefx.Tracer that allows the user to specify a set of leaf modules, ie.
    modules that are not to be traced through. The resulting graph ends up
    having single nodes referencing calls to the leaf modules' forward methods.
    """

    def __init__(self, *args, **kwargs):
        self.leaf_modules = {}
        if "leaf_modules" in kwargs:
            leaf_modules = kwargs.pop("leaf_modules")
            self.leaf_modules = leaf_modules
        super().__init__(*args, **kwargs)

    def is_leaf_module(self, m: nn.Module, module_qualname: str) -> bool:
        if isinstance(m, tuple(self.leaf_modules)):
            return True
        return super().is_leaf_module(m, module_qualname)


class NodePathTracer(LeafModuleAwareTracer):
    """
    NodePathTracer is an FX tracer that, for each operation, also records the
    name of the Node from which the operation originated. A node name here is
    a `.` separated path walking the hierarchy from top level module down to
    leaf operation or leaf module. The name of the top level module is not
    included as part of the node name. For example, if we trace a module whose
    forward method applies a ReLU module, the name for that node will simply
    be 'relu'.

    Some notes on the specifics:
        - Nodes are recorded to `self.node_to_qualname` which is a dictionary
          mapping a given Node object to its node name.
        - Nodes are recorded in the order which they are executed during
          tracing.
        - When a duplicate node name is encountered, a suffix of the form
          _{int} is added. The counter starts from 1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track the qualified name of the Node being traced
        self.current_module_qualname = ""
        # A map from FX Node to the qualified name\#
        self.node_to_qualname = OrderedDict()

    def call_module(self, m: oneflow.nn.Module, forward: Callable, args, kwargs):
        """
        Override of `onefx.Tracer.call_module`
        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Adds the qualified name of the caller to
           `current_module_qualname` for retrieval by `create_proxy`
        3) Once a leaf module is reached, calls `create_proxy`
        4) Restores the caller's qualified name into current_module_qualname
        """
        old_qualname = self.current_module_qualname
        try:
            module_qualname = self.path_of_module(m)
            self.current_module_qualname = module_qualname
            if not self.is_leaf_module(m, module_qualname):
                out = forward(*args, **kwargs)
                return out
            return self.create_proxy("call_module", module_qualname, args, kwargs)
        finally:
            self.current_module_qualname = old_qualname

    def create_proxy(
        self, kind: str, target: onefx.node.Target, args, kwargs, name=None, type_expr=None, *_
    ) -> onefx.proxy.Proxy:
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_qualname`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_qualname[proxy.node] = self._get_node_qualname(self.current_module_qualname, proxy.node)
        return proxy

    def _get_node_qualname(self, module_qualname: str, node: onefx.node.Node) -> str:
        node_qualname = module_qualname

        if node.op != "call_module":
            # In this case module_qualname from oneflow.fx doesn't go all the
            # way to the leaf function/op so we need to append it
            if len(node_qualname) > 0:
                # Only append '.' if we are deeper than the top level module
                node_qualname += "."
            node_qualname += str(node)

        # Now we need to add an _{index} postfix on any repeated node names
        # For modules we do this from scratch
        # But for anything else, oneflow.fx already has a globally scoped
        # _{index} postfix. But we want it locally (relative to direct parent)
        # scoped. So first we need to undo the oneflow.fx postfix
        if re.match(r".+_[0-9]+$", node_qualname) is not None:
            node_qualname = node_qualname.rsplit("_", 1)[0]

        # ... and now we add on our own postfix
        for existing_qualname in reversed(self.node_to_qualname.values()):
            # Check to see if existing_qualname is of the form
            # {node_qualname} or {node_qualname}_{int}
            if re.match(rf"{node_qualname}(_[0-9]+)?$", existing_qualname) is not None:
                postfix = existing_qualname.replace(node_qualname, "")
                if len(postfix):
                    # existing_qualname is of the form {node_qualname}_{int}
                    next_index = int(postfix[1:]) + 1
                else:
                    # existing_qualname is of the form {node_qualname}
                    next_index = 1
                node_qualname += f"_{next_index}"
                break

        return node_qualname


def _is_subseq(x, y):
    """Check if y is a subseqence of x
    https://stackoverflow.com/a/24017747/4391249
    """
    iter_x = iter(x)
    return all(any(x_item == y_item for x_item in iter_x) for y_item in y)


def _warn_graph_differences(train_tracer: NodePathTracer, eval_tracer: NodePathTracer):
    """
    Utility function for warning the user if there are differences between
    the train graph nodes and the eval graph nodes.
    """
    train_nodes = list(train_tracer.node_to_qualname.values())
    eval_nodes = list(eval_tracer.node_to_qualname.values())

    if len(train_nodes) == len(eval_nodes) and all(t == e for t, e in zip(train_nodes, eval_nodes)):
        return

    suggestion_msg = (
        "When choosing nodes for feature extraction, you may need to specify "
        "output nodes for train and eval mode separately."
    )

    if _is_subseq(train_nodes, eval_nodes):
        msg = (
            "NOTE: The nodes obtained by tracing the model in eval mode "
            "are a subsequence of those obtained in train mode. "
        )
    elif _is_subseq(eval_nodes, train_nodes):
        msg = (
            "NOTE: The nodes obtained by tracing the model in train mode "
            "are a subsequence of those obtained in eval mode. "
        )
    else:
        msg = "The nodes obtained by tracing the model in train mode are different to those obtained in eval mode. "
    warnings.warn(msg + suggestion_msg)


def _get_leaf_modules_for_ops() -> List[type]:
    # members = inspect.getmembers(flowvision.ops)
    result = []
    # for _, obj in members:
    #     if inspect.isclass(obj) and issubclass(obj, oneflow.nn.Module):
    #         result.append(obj)
    return result


def _set_default_tracer_kwargs(original_tr_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    default_autowrap_modules = (math, )
    default_leaf_modules = _get_leaf_modules_for_ops()
    result_tracer_kwargs = {} if original_tr_kwargs is None else original_tr_kwargs
    result_tracer_kwargs["autowrap_modules"] = (
        tuple(set(result_tracer_kwargs["autowrap_modules"] + default_autowrap_modules))
        if "autowrap_modules" in result_tracer_kwargs
        else default_autowrap_modules
    )
    result_tracer_kwargs["leaf_modules"] = (
        list(set(result_tracer_kwargs["leaf_modules"] + default_leaf_modules))
        if "leaf_modules" in result_tracer_kwargs
        else default_leaf_modules
    )
    return result_tracer_kwargs


def get_graph_node_names(
    model: nn.Module,
    tracer_kwargs: Optional[Dict[str, Any]] = None,
    suppress_diff_warning: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Dev utility to return node names in order of execution. See note on node
    names under :func:`create_feature_extractor`. Useful for seeing which node
    names are available for feature extraction. There are two reasons that
    node names can't easily be read directly from the code for a model:

        1. Not all submodules are traced through. Modules from ``oneflow.nn`` all
           fall within this category.
        2. Nodes representing the repeated application of the same operation
           or leaf module get a ``_{counter}`` postfix.

    The model is traced twice: once in train mode, and once in eval mode. Both
    sets of node names are returned.


    Args:
        model (nn.Module): model for which we'd like to print node names
        tracer_kwargs (dict, optional): a dictionary of keyword arguments for
            ``NodePathTracer`` (they are eventually passed onto
            `one-onefx.Tracer`_).
            By default it will be set to wrap and make leaf nodes all flowvision ops:
            {"autowrap_modules": (math, flowvision.ops,),"leaf_modules": _get_leaf_modules_for_ops(),}
            WARNING: In case the user provides tracer_kwargs, above default arguments will be appended to the user
            provided dictionary.

        suppress_diff_warning (bool, optional): whether to suppress a warning
            when there are discrepancies between the train and eval version of
            the graph. Defaults to False.

    Returns:
        tuple(list, list): a list of node names from tracing the model in
        train mode, and another from tracing the model in eval mode.

    Examples::

        >>> model = flowvision.models.resnet18()
        >>> train_nodes, eval_nodes = get_graph_node_names(model)
    """
    tracer_kwargs = _set_default_tracer_kwargs(tracer_kwargs)
    is_training = model.training
    train_tracer = NodePathTracer(**tracer_kwargs)
    train_tracer.trace(model.train())
    eval_tracer = NodePathTracer(**tracer_kwargs)
    eval_tracer.trace(model.eval())
    train_nodes = list(train_tracer.node_to_qualname.values())
    eval_nodes = list(eval_tracer.node_to_qualname.values())
    if not suppress_diff_warning:
        _warn_graph_differences(train_tracer, eval_tracer)
    # Restore training state
    model.train(is_training)
    return train_nodes, eval_nodes


class DualGraphModule(onefx.GraphModule):
    """
    A derivative of `onefx.GraphModule`. Differs in the following ways:
    - Requires a train and eval version of the underlying graph
    - Copies submodules according to the nodes of both train and eval graphs.
    - Calling train(mode) switches between train graph and eval graph.
    """

    def __init__(
        self, root: oneflow.nn.Module, train_graph: onefx.Graph, eval_graph: onefx.Graph, class_name: str = "GraphModule"
    ):
        """
        Args:
            root (nn.Module): module from which the copied module hierarchy is
                built
            train_graph (onefx.Graph): the graph that should be used in train mode
            eval_graph (onefx.Graph): the graph that should be used in eval mode
        """
        super(onefx.GraphModule, self).__init__()

        self.__class__.__name__ = class_name

        self.train_graph = train_graph
        self.eval_graph = eval_graph

        # Copy all get_attr and call_module ops (indicated by BOTH train and
        # eval graphs)
        for node in chain(iter(train_graph.nodes), iter(eval_graph.nodes)):
            if node.op in ["get_attr", "call_module"]:
                if not isinstance(node.target, str):
                    raise TypeError(f"node.target should be of type str instead of {type(node.target)}")
                _copy_attr(root, self, node.target)

        # train mode by default
        self.train()
        self.graph = train_graph

        # (borrowed from onefx.GraphModule):
        # Store the Tracer class responsible for creating a Graph separately as part of the
        # GraphModule state, except when the Tracer is defined in a local namespace.
        # Locally defined Tracers are not pickleable. This is needed because oneflow.package will
        # serialize a GraphModule without retaining the Graph, and needs to use the correct Tracer
        # to re-create the Graph during deserialization.
        if self.eval_graph._tracer_cls != self.train_graph._tracer_cls:
            raise TypeError(
                f"Train mode and eval mode should use the same tracer class. Instead got {self.eval_graph._tracer_cls} for eval vs {self.train_graph._tracer_cls} for train"
            )
        self._tracer_cls = None
        if self.graph._tracer_cls and "<locals>" not in self.graph._tracer_cls.__qualname__:
            self._tracer_cls = self.graph._tracer_cls

    def train(self, mode=True):
        """
        Swap out the graph depending on the selected training mode.
        NOTE this should be safe when calling model.eval() because that just
        calls this with mode == False.
        """
        # NOTE: Only set self.graph if the current graph is not the desired
        # one. This saves us from recompiling the graph where not necessary.
        if mode and not self.training:
            self.graph = self.train_graph
        elif not mode and self.training:
            self.graph = self.eval_graph
        return super().train(mode=mode)


def create_feature_extractor(
    model: nn.Module,
    return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
    train_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
    eval_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
    tracer_kwargs: Optional[Dict[str, Any]] = None,
    suppress_diff_warning: bool = False,
) -> onefx.GraphModule:
    """
    Creates a new graph module that returns intermediate nodes from a given
    model as dictionary with user specified keys as strings, and the requested
    outputs as values. This is achieved by re-writing the computation graph of
    the model via FX to return the desired nodes as outputs. All unused nodes
    are removed, together with their corresponding parameters.

    Desired output nodes must be specified as a ``.`` separated
    path walking the module hierarchy from top level module down to leaf
    operation or leaf module. For more details on the node naming conventions
    used here, please see the :ref:`relevant subheading <about-node-names>`
    in the `documentation`_.

    Not all models will be FX traceable, although with some massaging they can
    be made to cooperate. Here's a (not exhaustive) list of tips:

        - If you don't need to trace through a particular, problematic
          sub-module, turn it into a "leaf module" by passing a list of
          ``leaf_modules`` as one of the ``tracer_kwargs`` (see example below).
          It will not be traced through, but rather, the resulting graph will
          hold a reference to that module's forward method.
        - Likewise, you may turn functions into leaf functions by passing a
          list of ``autowrap_functions`` as one of the ``tracer_kwargs`` (see
          example below).
        - Some inbuilt Python functions can be problematic. For instance,
          ``int`` will raise an error during tracing. You may wrap them in your
          own function and then pass that in ``autowrap_functions`` as one of
          the ``tracer_kwargs``.

    Args:
        model (nn.Module): model on which we will extract the features
        return_nodes (list or dict, optional): either a ``List`` or a ``Dict``
            containing the names (or partial names - see note above)
            of the nodes for which the activations will be returned. If it is
            a ``Dict``, the keys are the node names, and the values
            are the user-specified keys for the graph module's returned
            dictionary. If it is a ``List``, it is treated as a ``Dict`` mapping
            node specification strings directly to output names. In the case
            that ``train_return_nodes`` and ``eval_return_nodes`` are specified,
            this should not be specified.
        train_return_nodes (list or dict, optional): similar to
            ``return_nodes``. This can be used if the return nodes
            for train mode are different than those from eval mode.
            If this is specified, ``eval_return_nodes`` must also be specified,
            and ``return_nodes`` should not be specified.
        eval_return_nodes (list or dict, optional): similar to
            ``return_nodes``. This can be used if the return nodes
            for train mode are different than those from eval mode.
            If this is specified, ``train_return_nodes`` must also be specified,
            and `return_nodes` should not be specified.
        tracer_kwargs (dict, optional): a dictionary of keyword arguments for
            ``NodePathTracer`` (which passes them onto it's parent class
            `one-onefx.Tracer`_).
            By default it will be set to wrap and make leaf nodes all flowvision ops:
            {"autowrap_modules": (math, flowvision.ops,),"leaf_modules": _get_leaf_modules_for_ops(),}
            WARNING: In case the user provides tracer_kwargs, above default arguments will be appended to the user
            provided dictionary.
        suppress_diff_warning (bool, optional): whether to suppress a warning
            when there are discrepancies between the train and eval version of
            the graph. Defaults to False.

    Examples::

        >>> # Feature extraction with resnet
        >>> model = flowvision.models.resnet18()
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> model = create_feature_extractor(
        >>>     model, {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = model(oneflow.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', oneflow.Size([1, 64, 56, 56])),
        >>>      ('feat2', oneflow.Size([1, 256, 14, 14]))]

        >>> # Specifying leaf modules and leaf functions
        >>> def leaf_function(x):
        >>>     # This would raise a TypeError if traced through
        >>>     return int(x)
        >>>
        >>> class LeafModule(oneflow.nn.Module):
        >>>     def forward(self, x):
        >>>         # This would raise a TypeError if traced through
        >>>         int(x.shape[0])
        >>>         return oneflow.nn.functional.relu(x + 4)
        >>>
        >>> class MyModule(oneflow.nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.conv = oneflow.nn.Conv2d(3, 1, 3)
        >>>         self.leaf_module = LeafModule()
        >>>
        >>>     def forward(self, x):
        >>>         leaf_function(x.shape[0])
        >>>         x = self.conv(x)
        >>>         return self.leaf_module(x)
        >>>
        >>> model = create_feature_extractor(
        >>>     MyModule(), return_nodes=['leaf_module'],
        >>>     tracer_kwargs={'leaf_modules': [LeafModule],
        >>>                    'autowrap_functions': [leaf_function]})

    """
    tracer_kwargs = _set_default_tracer_kwargs(tracer_kwargs)
    is_training = model.training

    if all(arg is None for arg in [return_nodes, train_return_nodes, eval_return_nodes]):

        raise ValueError(
            "Either `return_nodes` or `train_return_nodes` and `eval_return_nodes` together, should be specified"
        )

    if (train_return_nodes is None) ^ (eval_return_nodes is None):
        raise ValueError(
            "If any of `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified"
        )

    if not ((return_nodes is None) ^ (train_return_nodes is None)):
        raise ValueError("If `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified")

    # Put *_return_nodes into Dict[str, str] format
    def to_strdict(n) -> Dict[str, str]:
        if isinstance(n, list):
            return {str(i): str(i) for i in n}
        return {str(k): str(v) for k, v in n.items()}

    if train_return_nodes is None:
        return_nodes = to_strdict(return_nodes)
        train_return_nodes = deepcopy(return_nodes)
        eval_return_nodes = deepcopy(return_nodes)
    else:
        train_return_nodes = to_strdict(train_return_nodes)
        eval_return_nodes = to_strdict(eval_return_nodes)

    # Repeat the tracing and graph rewriting for train and eval mode
    tracers = {}
    graphs = {}
    mode_return_nodes: Dict[str, Dict[str, str]] = {"train": train_return_nodes, "eval": eval_return_nodes}
    for mode in ["train", "eval"]:
        if mode == "train":
            model.train()
        elif mode == "eval":
            model.eval()

        # Instantiate our NodePathTracer and use that to trace the model
        tracer = NodePathTracer(**tracer_kwargs)
        graph = tracer.trace(model)

        name = model.__class__.__name__ if isinstance(model, nn.Module) else model.__name__
        graph_module = onefx.GraphModule(tracer.root, graph, name)

        available_nodes = list(tracer.node_to_qualname.values())
        # FIXME We don't know if we should expect this to happen
        if len(set(available_nodes)) != len(available_nodes):
            raise ValueError(
                "There are duplicate nodes! Please raise an issue https://github.com/Oneflow-Inc/vision/issues"
            )
        # Check that all outputs in return_nodes are present in the model
        for query in mode_return_nodes[mode].keys():
            # To check if a query is available we need to check that at least
            # one of the available names starts with it up to a .
            if not any([re.match(rf"^{query}(\.|$)", n) is not None for n in available_nodes]):
                raise ValueError(
                    f"node: '{query}' is not present in model. Hint: use "
                    "`get_graph_node_names` to make sure the "
                    "`return_nodes` you specified are present. It may even "
                    "be that you need to specify `train_return_nodes` and "
                    "`eval_return_nodes` separately."
                )

        # Remove existing output nodes (train mode)
        orig_output_nodes = []
        for n in reversed(graph_module.graph.nodes):
            if n.op == "output":
                orig_output_nodes.append(n)
        if not orig_output_nodes:
            raise ValueError("No output nodes found in graph_module.graph.nodes")

        for n in orig_output_nodes:
            graph_module.graph.erase_node(n)

        # Find nodes corresponding to return_nodes and make them into output_nodes
        nodes = [n for n in graph_module.graph.nodes]
        output_nodes = OrderedDict()
        for n in reversed(nodes):
            module_qualname = tracer.node_to_qualname.get(n)
            if module_qualname is None:
                # NOTE - Know cases where this happens:
                # - Node representing creation of a tensor constant - probably
                #   not interesting as a return node
                # - When packing outputs into a named tuple like in InceptionV3
                continue
            for query in mode_return_nodes[mode]:
                depth = query.count(".")
                if ".".join(module_qualname.split(".")[: depth + 1]) == query:
                    output_nodes[mode_return_nodes[mode][query]] = n
                    mode_return_nodes[mode].pop(query)
                    break
        output_nodes = OrderedDict(reversed(list(output_nodes.items())))

        # And add them in the end of the graph
        with graph_module.graph.inserting_after(nodes[-1]):
            graph_module.graph.output(output_nodes)

        # Remove unused modules / parameters
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        # Keep track of the tracer and graph so we can choose the main one
        tracers[mode] = tracer
        graphs[mode] = graph

    # Warn user if there are any discrepancies between the graphs of the
    # train and eval modes
    if not suppress_diff_warning:
        _warn_graph_differences(tracers["train"], tracers["eval"])

    # Build the final graph module
    graph_module = DualGraphModule(model, graphs["train"], graphs["eval"], class_name=name)

    # Restore original training mode
    model.train(is_training)
    graph_module.train(is_training)

    return graph_module

def set_rng_seed(seed):
    oneflow.manual_seed(seed)
    random.seed(seed)
    
def get_available_models():
    return [
        k
        for k, v in flowvision.models.ModelCreator._model_list.items()
        if k[0].lower() == k[0] and k[0] != "_" and k != "get_weight" and k in flowvision.models.__dict__.keys()
    ]
    
class TestSubModule(oneflow.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = oneflow.nn.ReLU()

    def forward(self, x):
        x = x + 1
        x = x + 1
        x = self.relu(x)
        x = self.relu(x)
        return x


class TestModule(oneflow.nn.Module):
    def __init__(self):
        super().__init__()
        self.submodule = TestSubModule()
        self.relu = oneflow.nn.ReLU()

    def forward(self, x):
        x = self.submodule(x)
        x = x + 1
        x = x + 1
        x = self.relu(x)
        x = self.relu(x)
        return 

def leaf_function(x):
    return int(x)    

test_module_nodes = [
    "x",
    "submodule.add",
    "submodule.add_1",
    "submodule.relu",
    "submodule.relu_1",
    "add",
    "add_1",
    "relu",
    "relu_1",
]

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    Examples::
        >>> m = flowvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = flowvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(oneflow.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', oneflow.Size([1, 64, 56, 56])),
        >>>      ('feat2', oneflow.Size([1, 256, 14, 14]))]
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return 

class TestFxFeatureExtraction:
    inp = oneflow.rand(1, 3, 224, 224, dtype=oneflow.float32, device="cpu")
    model_defaults = {"num_classes": 1}
    leaf_modules = []

    def _create_feature_extractor(self, *args, **kwargs):
        """
        Apply leaf modules
        """
        tracer_kwargs = {}
        if "tracer_kwargs" not in kwargs:
            tracer_kwargs = {"leaf_modules": self.leaf_modules}
        else:
            tracer_kwargs = kwargs.pop("tracer_kwargs")
        return create_feature_extractor(*args, **kwargs, tracer_kwargs=tracer_kwargs, suppress_diff_warning=True)

    def _get_return_nodes(self, model):
        set_rng_seed(0)
        exclude_nodes_filter = [
            "getitem",
            "floordiv",
            "size",
            "chunk",
            "_assert",
            "eq",
            "dim",
            "getattr",
        ]
        train_nodes, eval_nodes = get_graph_node_names(
            model, tracer_kwargs={"leaf_modules": self.leaf_modules}, suppress_diff_warning=True
        )
        # Get rid of any nodes that don't return tensors as they cause issues
        # when testing backward pass.
        train_nodes = [n for n in train_nodes if not any(x in n for x in exclude_nodes_filter)]
        eval_nodes = [n for n in eval_nodes if not any(x in n for x in exclude_nodes_filter)]
        return random.sample(train_nodes, 10), random.sample(eval_nodes, 10)

    @pytest.mark.parametrize("model_name", get_available_models())
    def test_build_fx_feature_extractor(self, model_name):
        set_rng_seed(0)
        model = flowvision.models.__dict__[model_name](**self.model_defaults).eval()
        train_return_nodes, eval_return_nodes = self._get_return_nodes(model)
        # Check that it works with both a list and dict for return nodes
        self._create_feature_extractor(
            model, train_return_nodes={v: v for v in train_return_nodes}, eval_return_nodes=eval_return_nodes
        )
        self._create_feature_extractor(
            model, train_return_nodes=train_return_nodes, eval_return_nodes=eval_return_nodes
        )
        # Check must specify return nodes
        with pytest.raises(ValueError):
            self._create_feature_extractor(model)
        # Check return_nodes and train_return_nodes / eval_return nodes
        # mutual exclusivity
        with pytest.raises(ValueError):
            self._create_feature_extractor(
                model, return_nodes=train_return_nodes, train_return_nodes=train_return_nodes
            )
        # Check train_return_nodes / eval_return nodes must both be specified
        with pytest.raises(ValueError):
            self._create_feature_extractor(model, train_return_nodes=train_return_nodes)
        # Check invalid node name raises ValueError
        with pytest.raises(ValueError):
            # First just double check that this node really doesn't exist
            if not any(n.startswith("l") or n.startswith("l.") for n in chain(train_return_nodes, eval_return_nodes)):
                self._create_feature_extractor(model, train_return_nodes=["l"], eval_return_nodes=["l"])
            else:  # otherwise skip this check
                raise ValueError

    def test_node_name_conventions(self):
        model = TestModule()
        train_nodes, _ = get_graph_node_names(model)
        assert all(a == b for a, b in zip(train_nodes, test_module_nodes))

    @pytest.mark.parametrize("model_name", get_available_models())
    def test_forward_backward(self, model_name):
        model = flowvision.models.__dict__[model_name](**self.model_defaults).train()
        train_return_nodes, eval_return_nodes = self._get_return_nodes(model)
        model = self._create_feature_extractor(
            model, train_return_nodes=train_return_nodes, eval_return_nodes=eval_return_nodes
        )
        out = model(self.inp)
        out_agg = 0
        for node_out in out.values():
            if isinstance(node_out, Sequence):
                if len(node_out) == 0:
                    continue
                if not isinstance(node_out[0], oneflow.Tensor):
                    out_agg += sum(o for o in node_out if o is not None)
                else:
                    out_agg += sum(o.float().mean() for o in node_out if o is not None)
            elif isinstance(node_out, Mapping):
                out_agg += sum(o.float().mean() for o in node_out.values() if o is not None)
            else:
                # Assume that the only other alternative at this point is a Tensor
                out_agg += node_out.float().mean()
        out_agg.backward()

    def test_feature_extraction_methods_equivalence(self):
        model = flowvision.models.resnet18(**self.model_defaults).eval()
        return_layers = {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}

        ilg_model = IntermediateLayerGetter(model, return_layers).eval()
        fx_model = self._create_feature_extractor(model, return_layers)

        # Check that we have same parameters
        for (n1, p1), (n2, p2) in zip(ilg_model.named_parameters(), fx_model.named_parameters()):
            assert n1 == n2
            assert p1.equal(p2)

        # And that ouputs match
        with oneflow.no_grad():
            ilg_out = ilg_model(self.inp)
            fgn_out = fx_model(self.inp)
        assert all(k1 == k2 for k1, k2 in zip(ilg_out.keys(), fgn_out.keys()))
        for k in ilg_out.keys():
            assert ilg_out[k].equal(fgn_out[k])

    # @pytest.mark.parametrize("model_name", get_available_models())
    # currently fx for jit is not supported
    def test_jit_forward_backward(self, model_name):
        set_rng_seed(0)
        model = flowvision.models.__dict__[model_name](**self.model_defaults).train()
        train_return_nodes, eval_return_nodes = self._get_return_nodes(model)
        model = self._create_feature_extractor(
            model, train_return_nodes=train_return_nodes, eval_return_nodes=eval_return_nodes
        )
        model = oneflow.jit.script(model)
        fgn_out = model(self.inp)
        out_agg = 0
        for node_out in fgn_out.values():
            if isinstance(node_out, Sequence):
                out_agg += sum(o.float().mean() for o in node_out if o is not None)
            elif isinstance(node_out, Mapping):
                out_agg += sum(o.float().mean() for o in node_out.values() if o is not None)
            else:
                # Assume that the only other alternative at this point is a Tensor
                out_agg += node_out.float().mean()
        out_agg.backward()

    def test_train_eval(self):
        class TestModel(oneflow.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = oneflow.nn.Dropout(p=1.0)

            def forward(self, x):
                x = x.float().mean()
                x = self.dropout(x)  # dropout
                if self.training:
                    x += 100  # add
                else:
                    x *= 0  # mul
                x -= 0  # sub
                return x

        model = TestModel()

        train_return_nodes = ["dropout", "add", "sub"]
        eval_return_nodes = ["dropout", "mul", "sub"]

        def checks(model, mode):
            with oneflow.no_grad():
                out = model(oneflow.ones(10, 10))
            if mode == "train":
                # Check that dropout is respected
                assert out["dropout"].item() == 0
                # Check that control flow dependent on training_mode is respected
                assert out["sub"].item() == 100
                assert "add" in out
                assert "mul" not in out
            elif mode == "eval":
                # Check that dropout is respected
                assert out["dropout"].item() == 1
                # Check that control flow dependent on training_mode is respected
                assert out["sub"].item() == 0
                assert "mul" in out
                assert "add" not in out

        # Starting from train mode
        model.train()
        fx_model = self._create_feature_extractor(
            model, train_return_nodes=train_return_nodes, eval_return_nodes=eval_return_nodes
        )
        # Check that the models stay in their original training state
        assert model.training
        assert fx_model.training
        # Check outputs
        checks(fx_model, "train")
        # Check outputs after switching to eval mode
        fx_model.eval()
        checks(fx_model, "eval")

        # Starting from eval mode
        model.eval()
        fx_model = self._create_feature_extractor(
            model, train_return_nodes=train_return_nodes, eval_return_nodes=eval_return_nodes
        )
        # Check that the models stay in their original training state
        assert not model.training
        assert not fx_model.training
        # Check outputs
        checks(fx_model, "eval")
        # Check outputs after switching to train mode
        fx_model.train()
        checks(fx_model, "train")

    def test_leaf_module_and_function(self):
        class LeafModule(oneflow.nn.Module):
            def forward(self, x):
                # This would raise a TypeError if it were not in a leaf module
                int(x.shape[0])
                return oneflow.nn.functional.relu(x + 4)

        class TestModule(oneflow.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = oneflow.nn.Conv2d(3, 1, 3)
                self.leaf_module = LeafModule()

            def forward(self, x):
                leaf_function(x.shape[0])
                x = self.conv(x)
                return self.leaf_module(x)

        model = self._create_feature_extractor(
            TestModule(),
            return_nodes=["leaf_module"],
            tracer_kwargs={"leaf_modules": [LeafModule], "autowrap_functions": [leaf_function]},
        ).train()

        # Check that LeafModule is not in the list of nodes
        assert "relu" not in [str(n) for n in model.graph.nodes]
        assert "leaf_module" in [str(n) for n in model.graph.nodes]

        # Check forward
        out = model(self.inp)
        # And backward
        out["leaf_module"].float().mean().backward()

if __name__ == '__main__':
    with onefx.global_wrap(len, builtins):
        pytest.main()

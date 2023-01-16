"""
Modified from https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/split_utils.py
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

import onefx
from onefx.graph import map_arg
from .tools_common import NodeList
from onefx._compatibility import compatibility
from onefx.passes.utils import lift_subgraph_as_module, HolderModule

__all__ = ['getattr_recursive', 'setattr_recursive', 'Component', 'split_by_tags']

@compatibility(is_backward_compatible=False)
def getattr_recursive(obj, name):
    for layer in name.split("."):
        if hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj


@compatibility(is_backward_compatible=False)
def setattr_recursive(obj, attr, value):
    if "." not in attr:
        setattr(obj, attr, value)
    else:
        layer = attr.split(".")
        setattr_recursive(getattr(obj, layer[0]), ".".join(layer[1:]), value)


@compatibility(is_backward_compatible=False)
@dataclass
class Component:
    """
    A component serves as a container for a subgraph we want to create afterwards.
    """

    graph: onefx.Graph
    order: int
    name: str

    # Stores the placeholder nodes in `graph`.
    input_placeholders: List = field(default_factory=list)

    # Store the nodes in original graph that are placeholder in `graph`.
    orig_inputs: List = field(default_factory=list)

    # Store the nodes in original graph that are outputs in `graph`.
    orig_outputs: List = field(default_factory=list)

    # Mapping from get_attr node in original graph to get_attr node in `graph`.
    getattr_maps: Dict[onefx.Node, onefx.Node] = field(default_factory=dict)
    constructor_args: List[str] = field(default_factory=list)
    gm: Optional[onefx.GraphModule] = None


@compatibility(is_backward_compatible=False)
def split_by_tags(gm: onefx.GraphModule, tags: List[str]) -> onefx.GraphModule:
    """
    Splits a GraphModule using tags on its graph nodes. We honor the order of
    tags. For example, we have tags = ["a", "b", "c"], the function will create
    the initial submodules in the order of "a_0", "b_1", "c_2".

    To set a tag:
    gm.graph.nodes[idx].tag = "mytag"

    This will result in all nodes with the same tag being extracted and placed in their
    own submodule. For placeholder, output and get_attr node, the tag is ignored. placeholder
    and output nodes are created when needed while get_attr nodes get copied to submodules
    where they are used.

    Given the following module def:

    class SimpleModule(oneflow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = oneflow.nn.Linear(...)
            self.linear2 = oneflow.nn.Linear(...)
            self.linear3 = oneflow.nn.Linear(...)

        def forward(self, in1, in2):
            r1 = self.linear1(in1)
            r2 = self.linear2(in2)
            r3 = oneflow.cat([r1, r2])
            return self.linear3(r3)

    Marking the node corresponding to in1 with the tag sc.REQUEST_ONLY.lower() results in the following split:

    ro_0:
    def forward(self, in1):
        self = self.root
        linear1 = self.linear1(in1)
        return linear1

    main_1:
    def forward(self, in2, linear1):
        self = self.root
        linear2 = self.linear2(in2)
        cat_1 = oneflow.cat([linear1, linear2])
        linear3 = self.linear3(cat_1)
        return linear3

    main_0:
    def forward(self, in1, in2):
        self = self.root
        ro_0 = self.ro_0(in1)
        main_1 = self.main_1(in2, ro_0)
        return main_1
    """

    def flatten(x: onefx.node.Argument) -> NodeList:
        """
        Stores nodes in x to a list and returns the list.
        """
        r: NodeList = []
        map_arg(x, r.append)
        return r

    # Mapping from node in original module to node in created submodule.
    node_remapping: Dict[onefx.Node, onefx.Node] = {}

    # Mapping from node in original module or created submodules to
    # corresponding component.
    node_to_component: Dict[onefx.Node, Component] = {}

    # Mapping from tag to the corresponding component.
    tag_to_component: Dict[str, Component] = {}

    # Stores all components.
    all_components: List[Component] = []

    # Stores nodes that will be used in main graph.
    used_in_main: Dict[onefx.Node, None] = {}

    # Main graph after split.
    main_g = onefx.Graph()

    # Mapping from node in original module to node in main graph after split.
    main_remapping: Dict[onefx.Node, onefx.Node] = {}

    # Output node of original module.
    output_node: Optional[onefx.Node] = None

    # Create a component for each tag, we don't expect to create other components afterwards.
    for tag in tags:
        comp = Component(onefx.Graph(), len(all_components), f"{tag}")
        all_components.append(comp)
        tag_to_component[tag] = comp

    # Traverse the nodes in original graph and take care of them.
    for node in gm.graph.nodes:
        if node.op == "output":
            if output_node is not None:
                raise RuntimeError("Multiple output nodes in graph!")
            output_node = node
            continue

        # Placeholders in the original graph get copied to main graph.
        if node.op == "placeholder":
            main_remapping[node] = main_g.placeholder(node.name, type_expr=node.type)
            continue

        # Get_attr nodes are ignored because we are not tagging them.
        # Instead, we copy them directly to the submodules use them afterwards.
        if node.op == "get_attr":
            continue

        # Now we process callable nodes which are nodes with op of call_module,
        # call_function or call_method. Every callable nodes should be tagged.
        assert hasattr(node, "tag")

        upstream_components = [
            node_to_component[x]
            for x in flatten(node.args) + flatten(node.kwargs)
            if x.op not in {"placeholder", "get_attr"}
        ]

        comp = tag_to_component[node.tag]
        node_to_component[node] = comp

        # Max order of upperstream components.
        mx = max((c.order for c in upstream_components), default=0)

        # Expect the componet for `node` has higher order then its upstream components.
        assert comp.order >= mx

        # Map a input of `node` to nodes in the component's graph.
        def remap_func(x):
            # If input is a get_attr node, copy it to current component's graph.
            # Returns the get_attr node in current component's graph.
            if x.op == "get_attr":
                if x not in comp.getattr_maps:
                    comp.getattr_maps[x] = comp.graph.get_attr(
                        x.target, type_expr=x.type
                    )
                return comp.getattr_maps[x]

            # If input is not a placeholder, it should have been put into a component
            # already. If it's the current component then we return the corresponding
            # node in the component.
            if x.op != "placeholder" and node_to_component[x] == comp:
                return node_remapping[x]

            # If input is a placeholder or it's in other components, we want to make it
            # as a placeholder in current component's graph.
            if x not in comp.orig_inputs:
                comp.orig_inputs.append(x)
                comp.input_placeholders.append(
                    comp.graph.placeholder(x.name, type_expr=x.type)
                )
                used_in_main[x] = None

            return comp.input_placeholders[
                next(i for i, y in enumerate(comp.orig_inputs) if x is y)
            ]

        n = comp.graph.node_copy(node, remap_func)
        n.tag = node.tag  # type: ignore[attr-defined]
        node_remapping[node] = n
        node_to_component[n] = comp

    if output_node is None:
        raise RuntimeError("Graph had no output node!")

    for x in flatten(output_node.args[0]):
        if x.op == "get_attr":
            # We don't need components mapping for nodes of type "get_attr"
            # that are consumed by the output. Only need to make sure we create
            # corresponding counterparts in the resulting graph.
            main_remapping[x] = main_g.get_attr(x.name, type_expr=x.type)
        else:
            # All component results consumed by the output node should be
            # marked as "used in main".
            used_in_main[x] = None

    # If a node is used in main graph then we mark it as an output in the component
    # it belongs to.
    for n in used_in_main:
        if n.op != "placeholder":
            node_to_component[n].orig_outputs.append(n)

    # Now we create a graphmodule for each component.
    for comp in all_components:
        outs = tuple(map(node_remapping.__getitem__, comp.orig_outputs))

        # Take care of the args of FX output node. If there's a single
        # output then the output node args is like (output_single), else
        # if there're multiple outputs then the output node args is like
        # ((output_0, output_1, ...)).
        comp.graph.output(outs[0] if len(outs) == 1 else outs)

        comp.gm = lift_subgraph_as_module(gm, comp.graph)

        # Create a call_module node in main graph.
        main_node = main_g.call_module(
            comp.name,
            args=tuple(map(main_remapping.__getitem__, comp.orig_inputs)),
            kwargs=None,
        )

        if len(outs) == 1:
            main_remapping[comp.orig_outputs[0]] = main_node
        else:
            for i, o in enumerate(comp.orig_outputs):
                # Use Proxy to record getitem access.
                main_remapping[o] = onefx.Proxy(main_node)[i].node  # type: ignore[index]

    main_g.output(map_arg(output_node.args[0], main_remapping.__getitem__))
    main_root = HolderModule({comp.name: comp.gm for comp in all_components})

    # If the output nodes consumes get_attr directly in the original graph,
    # then we need to make sure get_attr is copied to the new graph.
    for x in flatten(output_node.args[0]):
        if x.op == "get_attr":
            setattr(main_root, x.name, getattr_recursive(gm, x.target))  # type: ignore[arg-type]

    return onefx.GraphModule(main_root, main_g)

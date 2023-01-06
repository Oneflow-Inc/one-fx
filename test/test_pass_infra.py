# Owner(s): ["module: fx"]

import oneflow
import sys
sys.path.append(r'../one-fx')
import fx
import unittest

from fx.passes.infra.pass_base import PassResult
from fx.passes.infra.pass_manager import (
    PassManager,
    this_before_that_pass_constraint,
    _topological_sort_passes,
)

def replace_add_with_mul_pass(gm):
    modified = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == oneflow._C.add:
            node.target = oneflow._C.mul
            modified = True
    return PassResult(gm, modified)

def replace_mul_with_div_pass(gm):
    modified = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == oneflow._C.mul:
            node.target = oneflow._C.div
            modified = True
    return PassResult(gm, modified)

class AddModule(oneflow.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = oneflow._C.add(x, x)
        z = oneflow._C.add(y, x)
        return z


class TestPassManager(unittest.TestCase):
    def test_pass_manager(self):
        """
        Tests that the pass manager runs the passes correctly.
        """

        m = AddModule()
        traced_m = fx.symbolic_trace(m)
        pm = PassManager(passes=[replace_add_with_mul_pass, replace_mul_with_div_pass], steps=5)

        pm.validate_constraints()
        self.assertEqual(len(pm.passes), 2)

        res = pm(traced_m)
        modified_m = res.graph_module
        assert isinstance(modified_m, fx.GraphModule)

        # Check that all call_function nodes are divs
        for node in modified_m.graph.nodes:
            if node.op == "call_function":
                self.assertEqual(node.target, oneflow._C.div)

    def test_this_before_that_pass_constraint(self):
        """
        Tests the construction of constraints
        """
        passes = [lambda x: 2 * x for _ in range(10)]
        pm = PassManager(passes)

        # add unfulfillable constraint
        pm.add_constraint(this_before_that_pass_constraint(passes[-1], passes[0]))

        with self.assertRaises(RuntimeError):
            pm.validate_constraints()


    def test_pass_manager_checks(self):
        """
        Tests that users can add in check functions correctly
        """
        m = AddModule()
        traced_m = fx.symbolic_trace(m)
        pm = PassManager(passes=[replace_add_with_mul_pass, replace_mul_with_div_pass])

        def check_div_target(graph_module):
            for node in graph_module.graph.nodes:
                if node.op == "call_function" and node.target != oneflow._C.div:
                    raise ValueError("Target should be div!")
        pm.add_checks(check_div_target)

        with self.assertRaises(ValueError):
            pm(traced_m)

    def test_pass_manager_bad_checks(self):
        """
        Checks that we error if we pass in a check function with the wrong parameters
        """
        def check_bad_args(graph_module, i):
            pass

        pm = PassManager()
        self.assertRaises(TypeError, pm.add_checks, check_bad_args)

    def test_topological_sort(self):
        """
        Tests that passes are correctly ordered based on contraints.
        """

        def pass0(x):
            return x

        def pass1(x):
            return x + 1

        def pass2(x):
            return x + 2

        def pass3(x):
            return x + 3

        def pass4(x):
            return x + 4

        def pass5(x):
            return x + 5

        # Not passing any constraints should keep the original order
        passes = [pass0, pass1, pass2, pass3, pass4, pass5]
        sorted = _topological_sort_passes(passes, [])
        self.assertEqual(sorted, passes)

        # Graph that we are constructing:
        #     5 ---->  0  <---- 4
        #     |                 |
        #     +-> 2 -> 3 -> 1 <-+
        # Which has a possible topological order of: [4, 5, 0, 2, 3, 1]
        passes = [pass0, pass1, pass2, pass3, pass4, pass5]
        constraints = [
            this_before_that_pass_constraint(pass5, pass0),
            this_before_that_pass_constraint(pass5, pass2),
            this_before_that_pass_constraint(pass4, pass0),
            this_before_that_pass_constraint(pass4, pass1),
            this_before_that_pass_constraint(pass2, pass3),
            this_before_that_pass_constraint(pass3, pass1),
        ]
        sorted = _topological_sort_passes(passes, constraints)
        self.assertEqual(sorted, [pass4, pass5, pass0, pass2, pass3, pass1])

        # Circular dependency should result in the circular_dep flag being set
        passes = [pass0, pass1, pass2]
        constraints = [
            this_before_that_pass_constraint(passes[0], passes[1]),
            this_before_that_pass_constraint(passes[1], passes[2]),
            this_before_that_pass_constraint(passes[2], passes[0]),
        ]
        with self.assertRaises(RuntimeError) as e:
            _topological_sort_passes(passes, constraints)
        expected_error_msg = f"Circular dependency detected within the following passes: {passes}"
        self.assertEqual(e.exception.args[0], expected_error_msg)

    def test_pass_manager_error(self):
        """
        Tests error catching + debug
        """
        def pass_fail(graph_module):
            raise RuntimeError("bad")

        m = AddModule()
        traced_m = fx.symbolic_trace(m)
        pm = PassManager(passes=[replace_add_with_mul_pass, replace_mul_with_div_pass, pass_fail])

        # Comment out this line to see the actual error message
        with self.assertRaisesRegex(Exception, "pass_fail"):
            pm(traced_m)

if __name__ == '__main__':
    unittest.main()
# Owner(s): ["module: fx"]

from __future__ import annotations    # type: ignore[attr-defined]
import oneflow
import typing
import sys
sys.path.append(r'../one-fx')
from onefx import symbolic_trace

class A:
    def __call__(self, x: oneflow.Tensor):
        return oneflow.add(x, x)

# No forward references
class M1(oneflow.nn.Module):
    def forward(self, x: oneflow.Tensor, a: A) -> oneflow.Tensor:
        return a(x)

# Forward references
class M2(oneflow.nn.Module):
    def forward(self, x: 'oneflow.Tensor', a: 'A') -> 'oneflow.Tensor':
        return a(x)

# Non-oneflow annotation with no internal forward references
class M3(oneflow.nn.Module):
    def forward(self, x: typing.List[oneflow.Tensor], a: A) -> oneflow.Tensor:
        return a(x[0])

# Non-oneflow annotation with internal forward references
class M4(oneflow.nn.Module):
    def forward(self, x: typing.List['oneflow.Tensor'], a: A) -> 'oneflow.Tensor':
        return a(x[0])

x = oneflow.rand(2, 3)

ref = oneflow.add(x, x)

traced1 = symbolic_trace(M1())
res1 = traced1(x, A())
assert oneflow.all(oneflow.eq(ref, res1))

traced2 = symbolic_trace(M2())
res2 = traced2(x, A())
assert oneflow.all(oneflow.eq(ref, res2))

traced3 = symbolic_trace(M3())
res3 = traced3([x], A())
assert oneflow.all(oneflow.eq(ref, res3))

traced4 = symbolic_trace(M4())
res4 = traced4([x], A())
assert oneflow.all(oneflow.eq(ref, res4))

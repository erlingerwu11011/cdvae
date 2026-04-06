import torch as th
from torch import Tensor, Size
from typing import Sequence
from zuko.lazy import LazyComposedTransform, Transform
from zuko.flows.autoregressive import MaskedAutoregressiveTransform, DependentTransform, MonotonicAffineTransform
from zuko.utils import unpack, broadcast

from common import Iota


class MaskedWidthAutoregressiveTransform(MaskedAutoregressiveTransform):

    def __init__(
        self,
        width: int = 8,
        shapes: Sequence[Size] = ((), ()),
        **kwargs,
    ):
        self.width = width
        width_shapes = [(*s, width) for s in shapes]

        super().__init__(
            shapes=width_shapes,
            univariate=MonotonicAffineTransform,
            **kwargs,
        )

    def meta(self, c: Tensor, x: Tensor) -> Transform:
        if c is not None:
            x = th.cat(broadcast(x, c, ignore=1), dim=-1)

        phi = self.hyper(x)
        phi = phi.unflatten(-1, (-1, self.total))
        phi = unpack(phi, self.shapes)
        phi = [phi_i.sum(dim=-1) for phi_i in phi]
        shift, scale = phi

        return DependentTransform(self.univariate(shift, scale), 1)


class ComposedMappedSolutionMapping(LazyComposedTransform):

    def __init__(
        self,
        iota: Iota,
        causal_ordered: bool = True,
        triangular: bool = True,
        **kwargs,
    ):
        self.iota = iota
        self.causal_ordered = causal_ordered
        self.triangular = triangular

        # Order
        order = arange = th.arange(iota.cardinality)
        if not causal_ordered:
            order = th.cat([iota.i(index) for index in iota.order[::-1]])
        self.causal_order = order

        # Orders for transform
        inverse = th.zeros_like(self.causal_order)
        inverse[self.causal_order] = arange.to(inverse.dtype)
        orders = [inverse, th.flipud(inverse)]

        transforms = [
            MaskedWidthAutoregressiveTransform(
                features=iota.cardinality,
                order=orders[i % 2] if not triangular else orders[0],
                **kwargs,
            ) for i in range(len(iota.order))
        ]

        super().__init__(*transforms)

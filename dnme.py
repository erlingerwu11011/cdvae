import torch as th
from torch import Tensor
from torch.distributions import Transform
from zuko.lazy import LazyTransform
from zuko.nn import MLP
from zuko.transforms import CouplingTransform, ComposedTransform, DependentTransform, MonotonicAffineTransform

from common import Iota


class DiagonalNoiseMechanism(LazyTransform):

    def __init__(
        self,
        iota: Iota,
        index: int,
        width: int = 8,
        causal_ordered: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.iota = iota
        self.index = index
        self.causal_ordered = causal_ordered

        # Dimensions and indexes
        if causal_ordered:
            in_indexes = iota.prefix(i=index)
        else:
            in_indexes = iota.suffix(i=index)
        self.register_buffer("in_indexes", in_indexes)
        self.register_buffer("out_indexes", iota.i(i=index))
        self.in_dim = self.in_indexes.size(0)
        self.out_dim = self.out_indexes.size(0)

        # Coupling mask
        coupling_mask = th.ones((iota.cardinality, )).bool()
        coupling_mask[self.out_indexes] = False
        self.register_buffer("coupling_mask", coupling_mask)

        # Hyper network
        self.width = width
        self.blocksize = self.out_dim * width
        out_features = self.blocksize * 2
        if self.in_dim > 0:
            self.g = MLP(self.in_dim, out_features, **kwargs)
        else:
            self.g = MLP(1, out_features, **kwargs)

    def meta(self, x: Tensor) -> Transform:
        if self.in_dim > 0:
            if self.causal_ordered:
                pa = x[..., :self.in_dim]
            else:
                pa = x[..., -self.in_dim:]
        else:
            pa = th.zeros_like(x[..., :1])

        param: th.Tensor = self.g(pa)

        # Diagonal log scale
        a = param[..., :self.blocksize]
        shape = (*a.shape[:-1], self.width, self.out_dim)
        a = a.reshape(shape)
        a = a.mean(dim=-2)

        # Bias
        b = param[..., self.blocksize:]
        shape = (*b.shape[:-1], self.width, self.out_dim)
        b = b.reshape(shape)
        b = b.mean(dim=-2)

        # Transform
        return DependentTransform(
            MonotonicAffineTransform(
                shift=b,
                scale=a,
            ),
            reinterpreted=1,
        )

    def forward(self) -> Transform:
        return CouplingTransform(self.meta, self.coupling_mask)


class DiagonalNoiseSolutionMapping(LazyTransform):

    def __init__(
        self,
        iota: Iota,
        causal_ordered: bool = True,
        triangular: bool = True,
        width: int = 8,
        **kwargs,
    ):
        super().__init__()

        assert triangular == True, "Conflict: Diagonal Noise solution mapping is always triangular."
        self.iota = iota
        self.causal_ordered = causal_ordered
        self.triangular = triangular

        # Dimension order
        order = th.arange(iota.cardinality)
        if not causal_ordered:
            order = th.cat([iota.i(index) for index in iota.order[::-1]])
        self.causal_order = order

        # Component order
        order = iota.order
        if not causal_ordered:
            order = iota.order[::-1]
        transforms = [
            DiagonalNoiseMechanism(
                iota=iota,
                index=i,
                width=width,
                causal_ordered=causal_ordered,
                **kwargs,
            ) for i in order
        ]
        self.transforms = th.nn.ModuleList(transforms)

    def forward(self) -> Transform:
        return ComposedTransform(*(t() for t in self.transforms))

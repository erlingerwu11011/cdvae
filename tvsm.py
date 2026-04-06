import torch as th
from torch import Tensor
from torch.distributions import Transform, constraints
from torchdiffeq import odeint
from typing import Callable, Union, Tuple
from zuko.lazy import LazyTransform
from zuko.nn import MLP

from common import Iota

ode_method = 'dopri5'


def change_ode_method(method: str):
    global ode_method
    ode_method = method


class FFJTransform(Transform):  # Copied from zuko
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        f: Callable[[Tensor, Tensor], Tensor],
        t0: Union[float, Tensor] = 0.0,
        t1: Union[float, Tensor] = 1.0,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.f = f
        self.t0 = t0
        self.t1 = t1
        self.atol = atol
        self.rtol = rtol
        self.trace_scale = 1e-2  # relax jacobian tolerances
        self.t_forward = th.linspace(self.t0, self.t1, 2)
        self.t_backward = th.linspace(self.t1, self.t0, 2)

    def _call(self, x: Tensor) -> Tensor:
        return odeint(
            func=self.f,
            y0=x,
            t=self.t_forward.to(x.device),
            atol=self.atol,
            rtol=self.rtol,
            method=ode_method,
        )[-1]

    @property
    def inv(self) -> Transform:
        return FFJTransform(
            f=self.f,
            t0=self.t1,
            t1=self.t0,
            atol=self.atol,
            rtol=self.rtol,
        )

    def _inverse(self, y: Tensor) -> Tensor:
        return odeint(
            func=self.f,
            y0=y,
            t=self.t_backward.to(y.device),
            atol=self.atol,
            rtol=self.rtol,
            method=ode_method,
        )[-1]

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        _, ladj = self.call_and_ladj(x)
        return ladj

    def call_and_ladj(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        eps = th.randn_like(x)

        def f_aug(t: Tensor, x_and_ladj: Tensor) -> Tensor:
            x, ladj = x_and_ladj
            with th.enable_grad():
                x = x.clone().requires_grad_()
                dx = self.f(t, x)
            (epsjp, ) = th.autograd.grad(dx, x, eps, create_graph=True)
            trace = (epsjp * eps).sum(dim=-1)

            return dx, trace * self.trace_scale

        ladj = th.zeros_like(x[..., 0])

        y, ladj = odeint(
            func=f_aug,
            y0=(x, ladj),
            t=self.t_forward.to(x.device),
            atol=self.atol,
            rtol=self.rtol,
            method=ode_method,
        )
        y, ladj = y[-1], ladj[-1]

        return y, ladj * (1 / self.trace_scale)


class TriangularVelocityFieldSolutionMapping(LazyTransform):

    def __init__(
        self,
        iota: Iota,
        causal_ordered: bool = True,
        triangular: bool = True,
        width: int = 8,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        **kwargs,
    ):
        super().__init__()

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

        # Triagular adjacency mask
        adjacency = (inverse[:, None] >= inverse)
        if not triangular:
            adjacency = th.ones_like(adjacency)
        self.register_buffer("adjacency", adjacency)

        # Hyper network for time
        kwargs.setdefault("activation", th.nn.Tanh)
        self.in_out_dim = iota.cardinality
        self.width = width
        self.blocksize = self.in_out_dim * width
        self.matsize = self.in_out_dim**2 * width
        self.tparams = MLP(1, self.matsize * 3 + self.blocksize, **kwargs)

        # Parameters for FFJTransform
        self.register_buffer("times", th.tensor((0.0, 1.0)))
        self.atol = atol
        self.rtol = rtol

        def g(x):
            t = th.tensor(1.).to(x.device)
            x = x[None, :]
            return self.f(t, x)[0]

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        params = self.tparams(t.reshape(1, 1))

        # Restructure: (w^T*x+b)^T*(u*sigmoid(g))
        params = params.reshape(-1)
        w = params[:self.matsize]
        w = w.reshape(self.width, self.in_out_dim, self.in_out_dim)
        u = params[self.matsize:2 * self.matsize]
        u = u.reshape(self.width, self.in_out_dim, self.in_out_dim)
        g = params[2 * self.matsize:3 * self.matsize]
        g = g.reshape(self.width, self.in_out_dim, self.in_out_dim)
        u = u * th.sigmoid(g)
        b = params[3 * self.matsize:]
        b = b.reshape(self.width, 1, self.in_out_dim)

        # Masking if triangle
        if self.triangular:
            w = w * self.adjacency
            u = u * self.adjacency

        # Simple masked linear
        x = th.unsqueeze(x, 0).repeat(self.width, 1, 1)
        h = th.tanh(th.matmul(x, w.transpose(-2, -1)) + b)
        dx = th.matmul(h, u.transpose(-2, -1)).mean(dim=0)
        return dx

    def forward(self) -> Transform:
        return FFJTransform(
            f=self.f,
            t0=self.times[0],
            t1=self.times[1],
            atol=self.atol,
            rtol=self.rtol,
        )

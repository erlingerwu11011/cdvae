import torch as th
from torch.distributions import Distribution
from zuko.lazy import LazyDistribution, Unconditional
from zuko.distributions import DiagNormal

from common import Iota


class ExogenousStandardNormal(LazyDistribution):

    def __init__(
        self,
        iota: Iota,
        markovian: bool = True,
        **kwargs,
    ):
        """Initialize Standard Normal distribution as exogenous distribution.

        Args:
            `iota` (Iota): Rules for vectorization by causal order.
            `markovian` (bool, optional): If exogenous variables are independent of each other. Defaults to True.
        """
        super().__init__()

        assert markovian == True, "Conflict: Standard Normal distribution is always Markovian."
        self.iota = iota
        self.markovian = True
        self.base = Unconditional(
            DiagNormal,
            th.zeros(self.iota.cardinality),
            th.ones(self.iota.cardinality),
            buffer=True,
        )

    def forward(self) -> Distribution:
        """Return Standard Normal distribution as exogenous distribution.

        Returns:
            Distribution: Standard Normal distribution as exogenous distribution.
        """
        return self.base()

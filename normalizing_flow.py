from torch.distributions import Distribution
from torch.nn import ModuleList
from zuko.distributions import Joint
from zuko.flows import MAF
from zuko.lazy import LazyDistribution

from common import Iota


class ExogenousNormalizingFlow(LazyDistribution):

    def __init__(
        self,
        iota: Iota,
        markovian: bool = True,
        transforms: int = 3,
        **kwargs,
    ):
        """Initialize Normalizing Flow as exogenous distribution.

        Args:
            `iota` (Iota): Rules for vectorization by causal order.
            `markovian` (bool, optional): If exogenous variables are independent of each other. Defaults to True.
            `transforms` (int, optional): Number of transformations in the Normalizing Flow. Defaults to 3.
        """
        super().__init__()
        self.iota = iota
        self.markovian = markovian
        self.transforms = transforms

        if self.markovian:
            self.flows = ModuleList([
                MAF(
                    features=self.iota.dimensions[i],
                    transforms=transforms,
                    **kwargs,
                ) for i in self.iota.order
            ])
        else:
            self.flow = MAF(
                features=self.iota.cardinality,
                transforms=transforms,
                **kwargs,
            )

    def forward(self) -> Distribution:
        """Return Normalizing Flow as exogenous distribution.

        Returns:
            Distribution: Normalizing Flow as exogenous distribution.
        """
        if self.markovian:
            return Joint(*[flow() for flow in self.flows])
        else:
            return self.flow()

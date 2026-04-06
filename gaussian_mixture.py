from torch.distributions import Distribution
from torch.nn import ModuleList
from zuko.distributions import Joint
from zuko.flows import GMM
from zuko.lazy import LazyDistribution

from common import Iota


class ExogenousGaussianMixture(LazyDistribution):

    def __init__(
        self,
        iota: Iota,
        markovian: bool = True,
        components: int = 4,
        **kwargs,
    ):
        """Initialize Gaussian Mixture model as exogenous distribution.

        Args:
            `iota` (Iota): Rules for vectorization by causal order.
            `markovian` (bool, optional): If exogenous variables are independent of each other. Defaults to True.
            `components` (int, optional): Number of components for each variable in the Gaussian Mixture model. Defaults to 2.
        """
        super().__init__()
        self.iota = iota
        self.markovian = markovian
        self.components = components

        if self.markovian:
            self.gmms = ModuleList([
                GMM(
                    features=self.iota.dimensions[i],
                    components=components,
                    **kwargs,
                ) for i in self.iota.order
            ])
        else:
            self.gmm = GMM(
                features=self.iota.cardinality,
                components=components * len(self.iota.order),
                **kwargs,
            )

    def forward(self) -> Distribution:
        """Return Gaussian Mixture model as exogenous distribution.

        Returns:
            Distribution: Gaussian Mixture model as exogenous distribution.
        """
        if self.markovian:
            return Joint(*[gmm() for gmm in self.gmms])
        else:
            return self.gmm()

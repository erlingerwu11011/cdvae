import torch
import torch.nn as nn


def safe_sqrt(x, epsilon=1e-12):
    """
    Compute the square root of input tensor `x` with numerical stability.

    Args:
        x (torch.Tensor): Input tensor.
        epsilon (float): Small constant to avoid division by zero or negative values.

    Returns:
        torch.Tensor: Square root of `x` with values clamped above `epsilon`.
    """
    return torch.sqrt(torch.clamp(x, min=epsilon))


def pdist2sq(X1, X2):
    """
    Compute the pairwise squared Euclidean distances between rows of two matrices.

    Args:
        X1 (torch.Tensor): First matrix of size (n1, d).
        X2 (torch.Tensor): Second matrix of size (n2, d).

    Returns:
        torch.Tensor: Pairwise squared distances of size (n1, n2).
    """
    X1_norm = (X1**2).sum(dim=1, keepdim=True)
    X2_norm = (X2**2).sum(dim=1, keepdim=True)
    dists = X1_norm + X2_norm.T - 2.0 * torch.mm(X1, X2.T)
    return torch.clamp(dists, min=0)  # Ensure no negative distances


def wasserstein(
    X, t, active_entries, weights=None, p=0.5, lam=10, its=10, sq=False, backpropT=True
):
    """
    Compute the Wasserstein distance between treatment groups for covariate balancing in causal inference.

    Args:
        X (torch.Tensor): Covariates matrix of size (n, d), where n is the number of samples and d is the feature dimension.
        t (torch.Tensor): Binary treatment assignment vector of size (n,), where values are 0 or 1.
        p (float): Target balance ratio between treatment groups (0 < p < 1).
        lam (float): Regularization parameter for scaling the distance matrix.
        weights (torch.Tensor, optional): Sample weights of size (n,). If None, equal weights are used.
        its (int): Number of iterations for the Sinkhorn algorithm.
        sq (bool): If True, use squared Euclidean distance; otherwise, use Euclidean distance.
        backpropT (bool): If False, stop gradients for transport matrix computation.

    Returns:
        torch.Tensor: Wasserstein distance.
        torch.Tensor: Regularized and scaled distance matrix.
    """
    assert (
        len(t.shape) == 2 and t.shape[1] == 1
    ), f"'t (treatment)' must have shape (n, 1). Got shape {t.shape} instead."

    t = t.view(X.size(0))
    # Indices of treated and control groups
    active_entries = active_entries.view(X.size(0))
    # Mask treatment and control groups based on active_entries
    active_indices = torch.where(active_entries > 0)[0]
    t_active = t[active_indices]
    X_active = X[active_indices]

    it = torch.where((t_active > 0) & (active_entries[active_indices] > 0))[0]
    ic = torch.where((t_active < 1) & (active_entries[active_indices] > 0))[0]
    Xt = X_active[it]
    Xc = X_active[ic]

    nc = float(Xc.size(0))
    nt = float(Xt.size(0))

    if nt < 10 or nc < 10:
        return 0

    if sq:
        M = pdist2sq(Xt, Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt, Xc))

    if weights is not None:
        Wc = weights[ic] / (weights[ic].sum() + 1e-7)
        Wt = weights[it] / (weights[it].sum() + 1e-7)
        Wtc_mask = Wt.unsqueeze(1) * Wc.unsqueeze(0)
        Wtc_mask = Wtc_mask.squeeze(-1)

    M_mean = (M * Wtc_mask).sum() if weights is not None else M.mean()
    delta = M.max().detach()
    eff_lam = (lam / M_mean).detach()

    # Compute new distance matrix with regularization
    row = delta * torch.ones((1, M.size(1)), device=M.device)
    col = torch.cat(
        [
            delta * torch.ones((M.size(0), 1), device=M.device),
            torch.zeros((1, 1), device=M.device),
        ],
        dim=0,
    )
    Mt = torch.cat([M, row], dim=0)
    Mt = torch.cat([Mt, col], dim=1)
    nt = int(Xt.size(0))
    nc = int(Xc.size(0))

    # Compute marginal vectors
    if weights is not None:
        a = torch.cat([p * Wt, (1 - p) * torch.ones((1, 1), device=X.device)], dim=0)
        b = torch.cat([(1 - p) * Wc, p * torch.ones((1, 1), device=X.device)], dim=0)
    else:
        a = torch.cat(
            [
                p * torch.ones((nt, 1), device=X.device) / nt,
                (1 - p) * torch.ones((1, 1), device=X.device),
            ],
            dim=0,
        )
        b = torch.cat(
            [
                (1 - p) * torch.ones((nc, 1), device=X.device) / nc,
                p * torch.ones((1, 1), device=X.device),
            ],
            dim=0,
        )

    # Compute kernel matrix
    Mlam = eff_lam * Mt
    K = torch.exp(-Mlam) + 1e-6  # Add constant to avoid NaN
    U = K * Mt
    a_zeros = (a == 0).float()
    a_corrected = (1 - a_zeros) * a + a_zeros * 1e-7
    ainvK = K / a_corrected

    # Sinkhorn iterations
    u = a.clone()
    for _ in range(its):
        u = 1.0 / (ainvK @ (b / ((u.T @ K).T)))

    v = b / ((u.T @ K).T)
    T = u * (v.T * K)

    if not backpropT:
        T = T.detach()

    # Compute Wasserstein distance
    E = T * Mt
    D = 2 * E.sum()

    return D


def deviance_loss(
    y_dist: torch.distributions.Distribution, y: torch.Tensor, y_dist_type: str
) -> torch.Tensor:
    eps = 1e-8

    if y_dist_type == "Negative Binomial":
        mean = y_dist.mean
        dispr = y_dist.total_count
        deviance = (
            y * torch.log(y + eps)
            - (y + dispr) * torch.log(y + dispr + eps)
            - y * torch.log(mean + eps)
            + (y + dispr) * torch.log(mean + dispr)
        )
        deviance = 2 * deviance

    if y_dist_type == "Poisson":
        mean = y_dist.mean
        Poisson_loss = nn.PoissonNLLLoss(log_input=False, full=True, reduction="none")
        deviance = Poisson_loss(mean, y)

    if y_dist_type == "continuous":
        mean = y_dist.mean
        std = y_dist.stddev
        deviance = (y - mean) ** 2
        deviance = deviance / (2 * std**2)

    return deviance


class GMMprior(nn.Module):
    def __init__(
        self,
        n_clusters,
        z_latent_dim,
        cov_type_p_z_given_c="diag",
        to_fix_pi_p_c="all",
        init_type_p_z_given_c="gmm",
        device="cpu",
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.z_latent_dim = z_latent_dim
        self.cov_type_p_z_given_c = cov_type_p_z_given_c
        self.to_fix_pi_p_c = to_fix_pi_p_c
        self.init_type_p_z_given_c = init_type_p_z_given_c
        self.device = device

        # Initialize prior parameters
        if self.to_fix_pi_p_c:
            self._pi_p_c = nn.Parameter(
                torch.ones(n_clusters, device=self.device) / n_clusters, requires_grad=False
            )
        else:
            self._pi_p_c = nn.Parameter(torch.ones(n_clusters, device=self.device) / n_clusters)

        self.mu_p_z_given_c = nn.Parameter(
            torch.zeros(z_latent_dim, n_clusters, device=self.device)
        )

        if cov_type_p_z_given_c == "diag":
            # (z_latent_dim, n_clusters)
            self._sigma_square_p_z_given_c = nn.Parameter(
                torch.ones(z_latent_dim, n_clusters, device=self.device)
            )

        elif cov_type_p_z_given_c == "full":
            ones = torch.ones(n_clusters, z_latent_dim, device=self.device)
            eye_mats = torch.diag_embed(ones)
            eye_mats = eye_mats.permute(1, 2, 0)  # (z_latent_dim, z_latent_dim, n_clusters)
            self._l_mat_p_z_given_c = nn.Parameter(
                eye_mats
            )  # covariance matrix of each p(z | c) (full covariance structure), initialized with identity matrix

    @property
    def pi_p_c(self):
        return torch.softmax(self._pi_p_c, dim=0)

    @property
    def sigma_square_p_z_given_c(self):
        if self.init_type_p_z_given_c == "gmm":
            return torch.nn.Softplus(beta=10)(self._sigma_square_p_z_given_c)
        elif self.init_type_p_z_given_c == "glorot":
            return torch.exp(self._sigma_square_p_z_given_c)

    @property
    def l_mat_p_z_given_c(self):
        if self.init_type_p_z_given_c == "gmm":
            d = torch.nn.Softplus(beta=10)(torch.diagonal(self._l_mat_p_z_given_c, dim1=0, dim2=1))
            d = torch.diag_embed(d).permute(1, 2, 0)
            mask = (
                torch.eye(d.shape[0], device=d.device)
                .unsqueeze(2)
                .repeat(1, 1, self._l_mat_p_z_given_c.shape[2])
            )
            l_mat_p_z_given_c = mask * d + (1 - mask) * self._l_mat_p_z_given_c

        elif self.init_type_p_z_given_c == "glorot":
            d = torch.exp(torch.diagonal(self._l_mat_p_z_given_c, dim1=0, dim2=1))
            d = torch.diag_embed(d).permute(1, 2, 0)
            mask = (
                torch.eye(d.shape[0]).unsqueeze(2).repeat(1, 1, self._l_mat_p_z_given_c.shape[2])
            )
            l_mat_p_z_given_c = mask * d + (1 - mask) * self._l_mat_p_z_given_c

        return l_mat_p_z_given_c

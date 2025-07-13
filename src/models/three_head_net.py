import torch
import torch.nn as nn


class Three_Head_Net(nn.Module):
    def __init__(
        self,
        dim_vitals: int,
        hidden_size: int,
        z_latent_dim: int,
        h_latent_dim: int,
        num_layers: int,
        rnn_dropout: float,
        y_dist_type: str,
    ):

        super().__init__()
        # (BS , seq_length, input_size )
        self.learn_rep = shared_rep(dim_vitals, hidden_size, h_latent_dim, num_layers, rnn_dropout)
        self.propensity_net = propensity_net(h_latent_dim)
        self.outcome_net = outcome_net(z_latent_dim, h_latent_dim, y_dist_type)

    def forward(
        self,
        x_prev: torch.Tensor,
        w_prev: torch.Tensor,
        y_prev: torch.Tensor,
        x: torch.Tensor,
        re: torch.Tensor,
    ) -> torch.Tensor:

        rep = self.learn_rep(x_prev, w_prev, y_prev, x)
        p_w = self.propensity_net(rep)
        mean_0, mean_1 = self.outcome_net(rep, re)

        return rep, p_w, mean_0, mean_1


class outcome_net(nn.Module):
    def __init__(self, z_latent_dim: int, h_latent_dim: int, y_dist_type: str):

        super().__init__()
        self.y_dist_type = y_dist_type

        self.learn_mean_1 = nn.Sequential(
            nn.Linear(z_latent_dim + h_latent_dim, (z_latent_dim + h_latent_dim) // 2),
            nn.LeakyReLU(),
            nn.Linear((z_latent_dim + h_latent_dim) // 2, 1),
        )

        self.learn_mean_0 = nn.Sequential(
            nn.Linear(z_latent_dim + h_latent_dim, (z_latent_dim + h_latent_dim) // 2),
            nn.LeakyReLU(),
            nn.Linear((z_latent_dim + h_latent_dim) // 2, 1),
        )

    def forward(self, rep: torch.Tensor, re: torch.Tensor) -> torch.Tensor:

        re = re.view(re.shape[0], 1, -1).repeat((1, rep.shape[1], 1))
        # ? normlize re and rep ?
        z_re_x = torch.cat((rep, re), dim=-1)
        mean_0 = self.learn_mean_0(z_re_x).view(z_re_x.shape[0], -1)
        mean_1 = self.learn_mean_1(z_re_x).view(z_re_x.shape[0], -1)
        if self.y_dist_type == "discrete":
            # Ensure positivity of Poisson parameter
            mean_0 = torch.exp(mean_0)
            mean_1 = torch.exp(mean_1)
        return mean_0, mean_1


class propensity_net(nn.Module):
    def __init__(self, h_latent_dim: int):

        super().__init__()
        # (BS , seq_length, input_size )
        self.net = nn.Sequential(
            nn.Linear(h_latent_dim, h_latent_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(h_latent_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        p_w = self.net(rep)
        p_w = p_w.view(p_w.shape[0], -1)
        return p_w


class shared_rep(nn.Module):
    def __init__(
        self,
        dim_vitals: int,
        hidden_size: int,
        h_latent_dim: int,
        num_layers: int,
        rnn_dropout: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dim_vitals + 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.map = nn.Sequential(
            nn.Linear(hidden_size + dim_vitals, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, h_latent_dim),
        )

    def forward(
        self, x_prev: torch.Tensor, w_prev: torch.Tensor, y_prev: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        y_prev = y_prev.unsqueeze(-1)
        w_prev = w_prev.unsqueeze(-1)

        data = torch.cat((x_prev, w_prev, y_prev), dim=-1)
        c, _ = self.lstm(data)

        c = torch.cat((c, x), dim=-1)
        rep = self.map(c)

        return rep

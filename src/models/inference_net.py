import torch
import torch.nn as nn


class Inference_Net(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        z_latent_dim: int,
        num_layer: int,
        dropout_rate: float,
        activation: int = "elu",
    ):
        super().__init__()
        self.activation = activation
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_size, z_latent_dim),
        )

        self.fc_var = nn.Sequential(
            nn.Linear(hidden_size, z_latent_dim),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        output, _ = self.gru(input)

        mu = self.fc_mu(output[:, -1, :])
        log_var = self.fc_var(output[:, -1, :])

        return mu, log_var, output

    def get_activation(self) -> nn.Module:
        if self.activation == "elu":
            return nn.ELU()
        elif self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        elif self.activation == "selu":
            return nn.SELU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

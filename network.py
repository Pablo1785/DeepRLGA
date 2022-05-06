import torch.nn as nn
import torch


class DQN(nn.Module):
    def __init__(self, num_state_features: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=num_state_features, out_features=24),
            nn.ReLU(),
            nn.Linear(in_features=24, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

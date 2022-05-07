import torch.nn as nn
import torch


class DQN(nn.Module):
    def __init__(self, num_state_features: int, num_outputs: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=num_state_features, out_features=64),
            nn.ReLU(),
            nn.Linear(64, 92),
            nn.ReLU(),
            nn.Linear(92, 24),
            nn.ReLU(),
            nn.Linear(24, 56),
            nn.ReLU(),
            nn.Linear(56, 42),
            nn.ReLU(),
            nn.Linear(in_features=42, out_features=num_outputs),
        )
        self.double()

    def forward(self, x: torch.Tensor):
        return self.net(x)

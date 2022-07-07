import torch.nn as nn
import torch


class DQN(nn.Module):
    def __init__(self, num_state_features: int, num_outputs: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=num_state_features, out_features=512),
            nn.ReLU(),
            nn.Linear(512, 360),
            nn.ReLU(),
            nn.Linear(360, 256),
            nn.ReLU(),
            nn.Linear(256, 168),
            nn.ReLU(),
            nn.Linear(168, 128),
            nn.ReLU(),
            nn.Linear(128, 92),
            nn.ReLU(),
            nn.Linear(in_features=92, out_features=num_outputs),
        )
        self.double()

    def forward(self, x: torch.Tensor):
        return self.net(x)

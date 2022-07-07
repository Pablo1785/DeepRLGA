import json
import random
from typing import Union

import torch

from deep_rl_ga.strategy import (
    EpsilonGreedyStrategy,
    NoExplorationStrategy,
)


class Agent:
    def __init__(
        self,
        strategy: Union[EpsilonGreedyStrategy, NoExplorationStrategy],
        num_actions: int,
        device,
    ):
        """

        :param strategy: Should implement 'get_exploration_rate() -> float' that returns a number in range [0,
        1] inclusive
        :param num_actions: Number of possible actions
        :param device: Pytorch device
        """
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(
        self,
        state: torch.Tensor,
        policy_net: torch.nn.Module,
    ):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)  # explore
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                network_output = policy_net(state)
                return network_output.unsqueeze(dim=0).argmax(dim=1).to(self.device)  # exploit

    def to_json(self):
        return json.dumps(
            {
                'strategy': None if type(self.strategy) == NoExplorationStrategy else self.strategy.to_json(),
                'num_actions': self.num_actions,
            }
        )
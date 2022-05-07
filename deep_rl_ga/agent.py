import random
import torch


class Agent:
    def __init__(
        self,
        strategy,
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
                return policy_net(state).unsqueeze(dim=0).argmax(dim=1).to(self.device)  # exploit
from collections import namedtuple
import random

import torch

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience: Experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


def extract_tensors(experiences, device):
    batch = Experience(*zip(*experiences))

    t1 = torch.nan_to_num(torch.stack(batch.state)).to(device)
    t2 = torch.nan_to_num(torch.cat(batch.action)).to(device)
    t3 = torch.nan_to_num(torch.cat(batch.reward)).to(device)
    t4 = torch.nan_to_num(torch.stack(batch.next_state)).to(device)

    return t1, t2, t3, t4
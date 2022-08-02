from collections import namedtuple
import random
from typing import (
    List,
    Tuple,
)

import numpy as np

import abc

import torch

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


class Memory(abc.ABC):
    def push(self, experience: Experience, **kwargs):
        raise NotImplementedError

    def sample(self, batch_size: int, include_latest = False):
        raise NotImplementedError

    def can_provide_sample(self, batch_size: int):
        raise NotImplementedError


class ReplayMemory(Memory):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def _get_latest_memory(self):
        if len(self.memory) < self.capacity:
            return self.memory[-1]
        return self.memory[(self.push_count - 1) % self.capacity]

    def push(self, experience: Experience, **kwargs):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size, include_latest=False):
        """
        Sample a random batch of transitions from memory buffer.

        :param batch_size: Number of samples to draw
        :param include_latest: Guarantee that the latest transition will be a part of the sample. This is called
        Combined Experience Replay, as in Zhang et al https://arxiv.org/pdf/1712.01275.pdf
        :return:
        """
        if include_latest:
            return random.sample(self.memory, batch_size - 1) + [self._get_latest_memory()]
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class PriorityReplayMemory(Memory):
    def __init__(self, capacity, e=0.01, a=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.push_count = 0
        self.e = e
        self.a = a
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def push(self, experience: Experience, **kwargs):
        """

        :param experience:
        :param kwargs: Should provide error value
        :return:
        """
        if 'error' not in kwargs:
            raise TypeError
        priority = self._get_priority(error=kwargs['error'])
        self.tree.add(priority, experience)
        self.push_count += 1

    def sample(self, batch_size, include_latest=False) -> Tuple[List[Experience], List[int], np.array]:
        """
        Sample a random batch of transitions from memory buffer.

        :param batch_size: Number of samples to draw
        :param include_latest: Guarantee that the latest transition will be a part of the sample. This is called
        Combined Experience Replay, as in Zhang et al https://arxiv.org/pdf/1712.01275.pdf
        :return: batch of samples, indices to easily access them from memory, weights of the samples
        """
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def can_provide_sample(self, batch_size):
        return self.push_count >= batch_size

    def update(
            self,
            idx,
            error
            ):
        p = self._get_priority(
            error
            )
        self.tree.update(
            idx,
            p
            )


def extract_tensors(experiences, device):
    batch = Experience(*zip(*experiences))

    t1 = torch.nan_to_num(torch.stack(batch.state)).to(device)
    t2 = torch.nan_to_num(torch.cat(batch.action)).to(device)
    t3 = torch.nan_to_num(torch.cat(batch.reward)).to(device)
    t4 = torch.nan_to_num(torch.stack(batch.next_state)).to(device)

    return t1, t2, t3, t4
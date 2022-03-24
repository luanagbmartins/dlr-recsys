import numpy as np
from src.model.tree import SumTree, MinTree
import random
import torch


class PriorityExperienceReplay(object):
    def __init__(self, buffer_size, embedding_dim, obs_size, device):
        self.device = device

        self.buffer_size = buffer_size
        self.crt_idx = 0
        self.is_full = False

        self.states = torch.zeros((buffer_size, obs_size), dtype=torch.float32).to(
            device
        )
        self.actions = torch.zeros(
            (buffer_size, embedding_dim), dtype=torch.float32
        ).to(device)
        self.rewards = torch.zeros((buffer_size), dtype=torch.float32).to(device)
        self.next_states = torch.zeros((buffer_size, obs_size), dtype=torch.float32).to(
            device
        )
        self.dones = torch.zeros(buffer_size, dtype=torch.bool).to(device)

        self.sum_tree = SumTree(buffer_size)
        self.min_tree = MinTree(buffer_size)

        self.max_priority = 1.0
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_constant = 0.00001

    def append(self, state, action, reward, next_state, done):
        self.states[self.crt_idx] = state
        self.actions[self.crt_idx] = action
        self.rewards[self.crt_idx] = reward
        self.next_states[self.crt_idx] = next_state
        self.dones[self.crt_idx] = done

        self.sum_tree.add_data(self.max_priority**self.alpha)
        self.min_tree.add_data(self.max_priority**self.alpha)

        self.crt_idx = (self.crt_idx + 1) % self.buffer_size
        if self.crt_idx == 0:
            self.is_full = True

    def sample(self, batch_size):
        rd_idx = []
        weight_batch = []
        index_batch = []
        sum_priority = self.sum_tree.sum_all_priority()

        N = self.buffer_size if self.is_full else self.crt_idx
        min_priority = self.min_tree.min_priority() / sum_priority
        max_weight = (N * min_priority) ** (-self.beta)

        segment_size = sum_priority / batch_size
        for j in range(batch_size):
            min_seg = segment_size * j
            max_seg = segment_size * (j + 1)

            random_num = random.uniform(min_seg, max_seg)
            priority, tree_index, buffer_index = self.sum_tree.search(random_num)
            rd_idx.append(buffer_index)

            p_j = priority / sum_priority
            w_j = (p_j * N) ** (-self.beta) / max_weight
            weight_batch.append(w_j)
            index_batch.append(tree_index)
        self.beta = min(1.0, self.beta + self.beta_constant)

        batch_states = self.states[rd_idx]
        batch_actions = self.actions[rd_idx]
        batch_rewards = self.rewards[rd_idx]
        batch_next_states = self.next_states[rd_idx]
        batch_dones = self.dones[rd_idx]

        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
            torch.FloatTensor(weight_batch).to(self.device),
            index_batch,
        )

    def update_priority(self, priority, index):
        self.sum_tree.update_priority(priority**self.alpha, index)
        self.min_tree.update_priority(priority**self.alpha, index)
        self.update_max_priority(priority**self.alpha)

    def update_max_priority(self, priority):
        self.max_priority = max(self.max_priority, priority)

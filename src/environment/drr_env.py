import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class OfflineEnv(object):
    def __init__(
        self,
        users_dict,
        users_history_lens,
        n_groups,
        item_groups,
        state_size,
        done_count,
        fairness_constraints,
        reward_threshold,
        reward_version,
        fix_user_id=None,
        reward_model=None,
        use_only_reward_model=False,
        device="cpu",
    ):

        self.device = device

        # users: interacted items, rate
        self.users_dict = users_dict
        self.users_history_lens = users_history_lens

        self.state_size = state_size

        self.reward_threshold = reward_threshold
        self.reward_version = reward_version

        # filter users with len_history > state_size
        self.available_users = self._generate_available_users()

        self.fix_user_id = fix_user_id

        self.user = (
            fix_user_id if fix_user_id else np.random.choice(self.available_users)
        )
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}
        self.items = [
            data[0]
            for data in self.users_dict[self.user]
            if data[1] >= reward_threshold
        ][: self.state_size]
        self.recommended_items = set(self.items)

        self.done = False
        self.done_count = done_count

        self.n_groups = n_groups
        self.item_groups = item_groups
        self.groups_movies = {
            i: [k for k in item_groups if item_groups[k] == i]
            for i in range(1, n_groups + 1)
        }
        self.group_count = {k: 0 for k in range(1, self.n_groups + 1)}
        self.total_recommended_items = 0
        self.fairness_constraints = fairness_constraints

        self.reward_model = reward_model
        self.use_only_reward_model = use_only_reward_model
        self.item_embeddings = (
            reward_model.item_embeddings.weight.data if reward_model else None
        )

    def _generate_available_users(self):
        available_users = []
        for u in self.users_dict.keys():
            positive_items = [
                data[0]
                for data in self.users_dict[u]
                if data[1] >= self.reward_threshold
            ]
            if len(positive_items) > self.state_size:
                available_users.append(u)
        return available_users

    def reward_normalization(self, reward):
        if self.reward_threshold <= 1:
            return reward

        if self.reward_model:
            return 0.5 * (reward - 3)
        else:
            return 1 if reward >= self.reward_threshold else -1

    def get_reward(self, action):
        reward = None

        if not self.use_only_reward_model:

            # If we know the reward according to the feedback log, we use it
            if (
                action in self.user_items.keys()
                and action not in self.recommended_items
            ):
                reward = self.reward_normalization(self.user_items[action])

            # If we dont know the reward, use reward predictor
            elif (
                action not in self.user_items.keys()
                and action not in self.recommended_items
            ):
                if self.reward_model:
                    reward = (
                        self.reward_model.predict(
                            torch.tensor([self.user]).long().to(self.device),
                            torch.tensor([action]).long().to(self.device),
                        )
                        .detach()
                        .cpu()
                        .numpy()[0]
                    )
                else:
                    reward = 0

        elif self.use_only_reward_model and (action not in self.recommended_items):
            reward = (
                self.reward_model.predict(
                    torch.tensor([self.user]).long().to(self.device),
                    torch.tensor([action]).long().to(self.device),
                )
                .detach()
                .cpu()
                .numpy()[0]
            )

        return reward

    def reset(self):
        self.user = (
            self.fix_user_id
            if self.fix_user_id
            else np.random.choice(self.available_users)
        )
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}
        self.items = [
            data[0]
            for data in self.users_dict[self.user]
            if data[1] >= self.reward_threshold
        ][: self.state_size]
        self.done = False
        self.recommended_items = set(self.items)
        self.correctly_recommended = set(self.items)
        self.group_count = {k: 0 for k in range(1, self.n_groups + 1)}
        self.total_recommended_items = 0
        return self.user, self.items, self.done

    def step(self, action, top_k=False):

        if top_k:
            precision = []
            _correctly_recommended, rewards = [], []
            for act in action:
                self.group_count[self.item_groups[act]] += 1
                self.total_recommended_items += 1

                _reward = self.get_reward(act)
                _reward = _reward if _reward else -1.5
                rewards.append(_reward)

                if _reward > 0:
                    _correctly_recommended.append(act)
                    self.correctly_recommended.add(act)
                self.recommended_items.add(act)
                precision.append(1 if _reward > 0 else 0)

            if max(precision) > 0:
                self.items = (
                    self.items[len(_correctly_recommended) :] + _correctly_recommended
                )
                self.items = self.items[-self.state_size :]
            reward = rewards

        else:
            precision = 0
            self.group_count[self.item_groups[action]] += 1
            self.total_recommended_items += 1

            reward = self.get_reward(action)
            reward = reward if reward else -1.5
            if reward > 0:
                self.items = self.items[1:] + [action]
                self.correctly_recommended.add(action)

            self.recommended_items.add(action)
            precision += 1 if reward > 0 else 0

        if self.total_recommended_items >= self.done_count:
            self.done = True

        return (
            self.items,
            reward,
            self.done,
            {"recommended_items": self.recommended_items, "precision": precision},
        )

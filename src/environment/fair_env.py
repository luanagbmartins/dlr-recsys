import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .drr_env import OfflineEnv


class OfflineFairEnv(OfflineEnv):
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
        super().__init__(
            users_dict,
            users_history_lens,
            n_groups,
            item_groups,
            state_size,
            done_count,
            fairness_constraints,
            reward_threshold,
            reward_version,
            fix_user_id,
            reward_model,
            use_only_reward_model,
            device,
        )

    def get_user_intent(self):
        user_intent = pd.DataFrame(
            self.item_embeddings[list(self.correctly_recommended)]
        )
        user_intent = cosine_similarity(user_intent, user_intent)[
            np.triu_indices(user_intent.shape[0], k=1)
        ]
        user_intent = (user_intent + 1) / 2
        user_intent = user_intent.mean() * (1 - user_intent.std())

        return user_intent

    def get_fair_reward(self, group, reward):

        if reward:

            if self.reward_version == "paper":
                # From the paper:
                if reward > 0:

                    fair_reward = (
                        (
                            self.fairness_constraints[group - 1]
                            / np.sum(self.fairness_constraints)
                        )
                        - (
                            self.group_count[group]
                            / np.sum(list(self.group_count.values()))
                        )
                        + 1
                    )
                else:
                    fair_reward = -1

            elif self.reward_version == "adaptative":
                # Adaptative:
                user_intent = self.get_user_intent()
                fair_reward = (
                    self.fairness_constraints[group - 1]
                    / np.sum(self.fairness_constraints)
                ) - (self.group_count[group] / np.sum(list(self.group_count.values())))
                fair_reward = fair_reward if user_intent <= 0.5 else reward

            elif self.reward_version == "combining":
                # Combining:
                user_intent = self.get_user_intent()
                fair_reward = (
                    self.fairness_constraints[group - 1]
                    / np.sum(self.fairness_constraints)
                ) - (self.group_count[group] / np.sum(list(self.group_count.values())))
                fair_reward = (reward * user_intent) + (fair_reward * (1 - user_intent))

        else:
            fair_reward = -1.5

        return fair_reward

    def step(self, action, top_k=False):

        if top_k:
            correctly_recommended = []
            rewards = []
            precision = []
            for act in action:
                group = self.item_groups[act]
                self.group_count[group] += 1
                self.total_recommended_items += 1

                _reward = self.get_reward(act)
                _reward = _reward if _reward else -1.5
                _fair_reward = self.get_fair_reward(group, self.get_reward(act))
                rewards.append(_fair_reward)

                if _reward > 0:
                    correctly_recommended.append(act)
                    self.correctly_recommended.add(act)

                self.recommended_items.add(act)
                precision.append(1 if _reward > 0 else 0)

            if max(precision) > 0:
                self.items = (
                    self.items[len(correctly_recommended) :] + correctly_recommended
                )
                self.items = self.items[-self.state_size :]
            reward = rewards

        else:
            group = self.item_groups[action]
            self.group_count[group] += 1
            self.total_recommended_items += 1

            _reward = self.get_reward(action)
            _reward = _reward if _reward else -1.5
            _fair_reward = self.get_fair_reward(group, self.get_reward(action))

            if _reward > 0:
                self.items = self.items[1:] + [action]
                self.correctly_recommended.add(action)

            self.recommended_items.add(action)
            precision = 1 if _reward > 0 else 0

            reward = _fair_reward

        if self.total_recommended_items >= self.done_count:
            self.done = True

        return (
            self.items,
            reward,
            self.done,
            {"recommended_items": self.recommended_items, "precision": precision},
        )

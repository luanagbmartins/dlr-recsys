import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class OfflineEnv(object):
    def __init__(
        self,
        users_dict,
        n_groups,
        item_groups,
        state_size,
        done_count,
        reward_threshold,
        fix_user_id=None,
        reward_model=None,
        use_only_reward_model=False,
        device="cpu",
        **kwargs
    ):
        """Offline environment for recommendation system.


        ### Action Space
        Items to be recommended to the user, in range [0, n_items - 1].


        ### Observation Space
        DRR-ave based on Liu et al. (2018)

        ### Rewards
        Float in range [-1, 1].


        Parameters
        ----------

        users_dict: dictionary
            User dictionary with interacted items and rate.
            Format: {user_id: [(item_id, rate), ...]}

        n_groups: int
            Number of groups of items (for fairness metrics).

        item_groups: dictionary
            Dictionary of item groups (for fairness metrics).
            Format: {item_id: group_id}

        state_size: int
            Number of items considered in the state.

        done_count: int
            Length of the episode.

        reward_threshold: int
            threshold for considering item rating a positive reward.

        fix_user_id: int
            User id. If set, the environment will always return the same user.

        reward_model: torch model
            reward predictor model (if not provided, the environment will use the feedback log).

        use_only_reward_model: boolean
            If set, the environment will only use the reward predictor model.


        References
        ----------
        Liu, F., Tang, R., Li, X., Ye, Y., Chen, H., Guo, H., & Zhang, Y. (2018).
        Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling.
        https://arxiv.org/pdf/1810.12027.pdf

        """

        self.device = device

        self.users_dict = users_dict
        self.state_size = state_size
        self.reward_threshold = reward_threshold
        self.reward_model = reward_model
        self.use_only_reward_model = use_only_reward_model
        self.fix_user_id = fix_user_id
        self.done_count = done_count
        self.n_groups = n_groups
        self.item_groups = item_groups

        # filter users with len_history > state_size
        self.available_users = self._generate_available_users()

        # if not fix_user_id choose a random user
        self.user = (
            fix_user_id if fix_user_id else np.random.choice(self.available_users)
        )
        # user items and ratings
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}

        # get last state_size items positively rated by the user in the feedback log
        self.items = [
            data[0]
            for data in self.users_dict[self.user]
            if data[1] >= reward_threshold
        ][: self.state_size]
        self.recommended_items = set(self.items)
        self.correctly_recommended = set(self.items)

        # get list of items in each group
        self.groups_items = {
            i: [k for k in item_groups if item_groups[k] == i]
            for i in range(1, n_groups + 1)
        }

        # initialize group count
        self.group_count = {k: 0 for k in range(1, self.n_groups + 1)}
        self.total_recommended_items = 0

        self.done = False

    def _generate_available_users(self):
        """Generate a list of available users.
        Get all users with at least state_size items positively rated in the feedback log.

        Returns
        -------
        available_users: array
            An array of available users.
        """

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

    def _reward_normalization(self, reward):
        """Normalize reward to [-1, 1]

        Parameters
        ----------
        reward: float
            Reward to be normalized.

        Returns
        -------
        reward: float
            Reward normalized to [-1, 1].
        """

        if self.reward_threshold <= 1:
            return reward

        if self.reward_model:
            return 0.5 * (reward - 3)
        else:
            return 1 if reward >= self.reward_threshold else -1

    def _get_reward(self, action):
        """Get reward for the action.
        If use_only_reward_model is set, the environment will only use the reward predictor model.
        Otherwise, the environment will use the feedback log if we know the reward according to the feedback log,
        otherwise, the environment will use the reward predictor model.

        Parameters
        ----------
        action: int
            Action to be evaluated.

        Returns
        -------
        reward: float
            Reward for the action.
        """

        reward = None
        if not self.use_only_reward_model:

            # If we know the reward according to the feedback log, we use it
            if (
                action in self.user_items.keys()
                and action not in self.recommended_items
            ):
                reward = self._reward_normalization(self.user_items[action])

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

    def get_recommended_items(self):
        """Get recommended items.
        Get the recommended items for the current user.

        Returns
        ----------
        recommended_items: array
            Array of items recommended for the current user.
        """

        return self.recommended_items

    def get_group_count(self):
        """Get group count.
        Get the number of items recommended in each group.

        Returns
        ----------
        group_count: array
            Array of items recommended in each group.
        """
        return list(self.group_count.values())

    def reset(self):
        """Reset the environment.

        Get a new user and reset the state.

        Returns
        -------
        user: int
            Next user id
        items: array
            last state_size items positively rated by the user in the feedback log
        done: bool
            If the episode is done
        """

        # if not fix_user_id choose a random user
        self.user = (
            self.fix_user_id
            if self.fix_user_id
            else np.random.choice(self.available_users)
        )

        # user items and ratings
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}

        # get last state_size items positively rated by the user in the feedback log
        self.items = [
            data[0]
            for data in self.users_dict[self.user]
            if data[1] >= self.reward_threshold
        ][: self.state_size]
        self.recommended_items = set(self.items)
        self.correctly_recommended = set(self.items)

        # reset group count
        self.group_count = {k: 0 for k in range(1, self.n_groups + 1)}
        self.total_recommended_items = 0

        self.done = False
        return self.user, self.items, self.done

    def step(self, action, top_k=False):
        """Step the environment.
        Receives the chosen action and calculates the reward assigned to that action

        Parameters
        ----------
        action: int or array
            Action to be evaluated.
        top_k: bool
            If True, action is a array of top k recommended items, otherwise action is a single item.

        Returns
        ----------
        next_items_ids: array
            Next state of the environment.
        reward: float
            Reward for the action.
        done: bool
            If the episode is done
        info: dictonary
            Extra information about the environment.
            Format: {'recommended_items': recommended_items, 'precision': precision}
        """

        if top_k:
            precision = []
            correctly_recommended, rewards = [], []
            for act in action:
                self.group_count[self.item_groups[act]] += 1
                self.total_recommended_items += 1

                _reward = self._get_reward(act)
                _reward = _reward if _reward else -1.5
                rewards.append(_reward)

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
            precision = 0
            self.group_count[self.item_groups[action]] += 1
            self.total_recommended_items += 1

            reward = self._get_reward(action)
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

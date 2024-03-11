import numpy as np
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from .drr_env import OfflineEnv


class OfflineFairEnv(OfflineEnv):
    def __init__(
        self,
        users_dict,
        n_groups,
        item_groups,
        items_metadata,
        items_df,
        state_size,
        done_count,
        fairness_constraints,
        reward_threshold,
        reward_version,
        user_intent,
        user_intent_threshold,
        title_emb_path=None,
        fix_user_id=None,
        reward_model=None,
        use_only_reward_model=False,
        device="cpu",
    ):
        """Offline fair environment for recommendation system.

        ### Action Space
        Items to be recommended to the user, in range [0, n_items - 1].

        ### Observation Space
        DRR-ave based on Liu et al. (2020)

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

        items_metadata: dataframe
            Dataframe with item metadata.

        items_df: dataframe
            Dataframe with items.

        state_size: int
            Number of items considered in the state.

        done_count: int
            Length of the episode.

        fairness_constraints: list
            List of fairness constraints. (weight of each group in the fairness metric)

        reward_threshold: int
            threshold for considering item rating a positive reward.

        reward_version: string
            Version of the reward function.

        user_intent: string
            Version of the user intent function.

        user_intent_threshold: float
            Threshold for considering user intent a positive reward.

        fix_user_id: int
            User id. If set, the environment will always return the same user.

        reward_model: torch model
            reward predictor model (if not provided, the environment will use the feedback log).

        use_only_reward_model: boolean
            If set, the environment will only use the reward predictor model.


        References
        ----------
        Liu, W., Liu, F., Tang, R., Liao, B., Chen, G., & Heng, P. (2020).
        Balancing Between Accuracy and Fairness for Interactive Recommendation with Reinforcement Learning.
        Advances in Knowledge Discovery and Data Mining, 12084, 155 - 167.
        https://arxiv.org/pdf/2106.13386.pdf


        """
        super().__init__(
            users_dict=users_dict,
            n_groups=n_groups,
            item_groups=item_groups,
            state_size=state_size,
            done_count=done_count,
            reward_threshold=reward_threshold,
            fix_user_id=fix_user_id,
            reward_model=reward_model,
            use_only_reward_model=use_only_reward_model,
            device=device,
        )

        self.reward_version = reward_version
        self.user_intent_threshold = user_intent_threshold
        self.fairness_constraints = fairness_constraints
        self.user_intent = user_intent
        self.items_metadata = items_metadata
        self.items_df = items_df

        # self.item_embeddings = (
        #     torch.tensor(self.reward_model.item_features_).to(self.device)
        #     if self.reward_model
        #     else None
        # )

        self.item_embeddings = None
        # (self.emb_model.item_embeddings.weight.data if self.reward_model else None)

        self.items_metadata = self.items_metadata.set_index("item_id")
        self.items_metadata = self.items_metadata[["metadata"]]

        self.title_emb = None
        if ("item_name" in self.user_intent) and title_emb_path:
            self.title_emb = pd.read_csv(title_emb_path)

    def _get_user_intent(self):
        """Get user intent.

        Returns
        -------
        user_intent: float
            User intent.

        Notes
        -------
        User intent is the cosine similarity between all items positively rated by the user.
        If user_intent = 1 the user likes similar items.
        If user_intent = 0 the user likes different items.
        """

        if self.user_intent == "item_emb_pmf":  # Item embedding PMF
            user_intent = pd.DataFrame(
                self.item_embeddings[list(self.correctly_recommended)].cpu().numpy()
            )
            user_intent = cosine_similarity(user_intent, user_intent)[
                np.triu_indices(user_intent.shape[0], k=1)
            ]
            user_intent = (user_intent + 1) / 2
            user_intent = user_intent.mean()

        elif (
            self.user_intent == "item_metadata_emb"
        ):  # where metadata is onehot encoded
            user_intent = self.items_metadata[
                self.items_metadata.index.isin(list(self.correctly_recommended))
            ]
            user_intent = user_intent.drop(columns=["item_id", "metadata"], axis=1)
            user_intent = cosine_similarity(user_intent, user_intent)[
                np.triu_indices(user_intent.shape[0], k=1)
            ]
            user_intent = (user_intent + 1) / 2
            user_intent = user_intent.mean()

        elif self.user_intent == "item_name_emb":  # Item name bert embedding
            _items_df = self.items_df[
                self.items_df["item_id"].isin(list(self.correctly_recommended))
            ]

            user_intent = self.title_emb.loc[_items_df["item_id"].tolist()]
            user_intent = cosine_similarity(user_intent, user_intent)[
                np.triu_indices(user_intent.shape[0], k=1)
            ]
            user_intent = (user_intent + 1) / 2
            user_intent = user_intent.mean()

        elif (
            self.user_intent == "item_name_metadata_emb"
        ):  # Item name bert embedding + metadata
            _items_df = self.items_df[
                self.items_df["item_id"].isin(list(self.correctly_recommended))
            ]

            user_intent = self.items_metadata[
                self.items_metadata.index.isin(list(self.correctly_recommended))
            ]
            user_intent = user_intent.drop(["metadata"], axis=1).values
            sentence_embeddings = self.title_emb.loc[_items_df["item_id"].tolist()]

            user_intent = np.concatenate((user_intent, sentence_embeddings), axis=1)
            user_intent = cosine_similarity(user_intent, user_intent)[
                np.triu_indices(user_intent.shape[0], k=1)
            ]
            user_intent = (user_intent + 1) / 2
            user_intent = user_intent.mean()

        else:
            raise "Not valid user intent"
        return 1 - user_intent

    def _get_fair_reward(self, group, reward):
        """Get reward for the action.

        fair_reward = (optimal_exposure - real_exposure) + 1
        reward = user_feedback

        ### Paper
        If user likes the item, the reward is:
            fair_reward
        Otherwise, the reward is:
            penalty (-1)

        ### Adaptative
        If user likes similar items, the reward is:
            reward

        If user likes different items, and the user likes the item,
        the reward is:
            fair_reward
        Otherwise, the reward is:
            fair_reward -1

        ### Combining
        Reward is the combination of the two rewads.
            (reward * user_intent) + (fair_reward * (1 - user_intent))

        """

        if reward:
            if self.reward_version == "paper":
                # From the paper:
                if reward > 0:

                    fair_reward = (
                        (
                            self.fairness_constraints[group - 1]
                            / np.sum(self.fairness_constraints)
                        )
                        - (self.group_count[group] / np.sum(self.get_group_count()))
                        + 1
                    )
                else:
                    fair_reward = -1

            # elif self.reward_version == "adaptative":
            #     # Adaptative:
            #     user_intent = self._get_user_intent()
            #     fair_reward = (
            #         self.fairness_constraints[group - 1]
            #         / np.sum(self.fairness_constraints)
            #     ) - (self.group_count[group] / np.sum(self.get_group_count()))

            #     fair_reward = (
            #         fair_reward if user_intent <= self.user_intent_threshold else reward
            #     )

            elif self.reward_version == "combining":
                # Combining:
                user_intent = self._get_user_intent()
                fair_reward = (
                    self.fairness_constraints[group - 1]
                    / np.sum(self.fairness_constraints)
                ) - (self.group_count[group] / np.sum(self.get_group_count()))

                fair_reward = (reward * (1 - user_intent)) + (
                    fair_reward * (user_intent)
                )

        else:
            # The item is already recommended, therefore penalizes the agent
            fair_reward = -1.5

        return fair_reward

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
            correctly_recommended = []
            rewards = []
            precision = []
            for act in action:
                group = self.item_groups[act]
                self.group_count[group] += 1
                self.total_recommended_items += 1

                _reward = self._get_reward(act)
                _fair_reward = self._get_fair_reward(group, _reward)
                _reward = _reward if _reward else -1.5
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

            _reward = self._get_reward(action)
            _fair_reward = self._get_fair_reward(group, _reward)
            _reward = _reward if _reward else -1.5

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

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
        items_metadata,
        items_df,
        state_size,
        done_count,
        fairness_constraints,
        reward_threshold,
        reward_version,
        user_intent,
        user_intent_threshold,
        fix_user_id=None,
        reward_model=None,
        use_only_reward_model=False,
        device="cpu",
    ):
        super().__init__(
            users_dict=users_dict,
            users_history_lens=users_history_lens,
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
        # self.items_df = self.items_df[["item_id", "title"]]

        self.items_metadata = self.items_metadata.set_index("item_id")
        filter_col = [col for col in self.items_metadata]
        self.items_metadata["genres"] = (
            self.items_metadata[filter_col]
            .dot(pd.Index(filter_col) + ", ")
            .str.strip(", ")
        )

        self.bert = None

        # if self.user_intent == "item_title_emb":
        #     from sentence_transformers import SentenceTransformer

        #     self.bert = SentenceTransformer("bert-base-nli-mean-tokens")

    def get_user_intent(self):
        if self.user_intent == "item_emb_pmf":
            user_intent = pd.DataFrame(
                self.item_embeddings[list(self.correctly_recommended)].cpu().numpy()
            )
            user_intent = cosine_similarity(user_intent, user_intent)[
                np.triu_indices(user_intent.shape[0], k=1)
            ]
            user_intent = (user_intent + 1) / 2
            user_intent = user_intent.mean() * (1 - user_intent.std())

        elif self.user_intent == "item_genre_emb":
            user_intent = self.items_metadata[
                self.items_metadata.index.isin(list(self.correctly_recommended))
            ]
            user_intent = user_intent.drop(columns=["item_id", "genres"], axis=1)
            user_intent = cosine_similarity(user_intent, user_intent)[
                np.triu_indices(user_intent.shape[0], k=1)
            ]
            user_intent = (user_intent + 1) / 2
            user_intent = user_intent.mean() * (1 - user_intent.std())

        elif self.user_intent == "item_genre_emb_bert":
            _items_df = self.items_metadata[
                self.items_metadata.index.isin(list(self.correctly_recommended))
            ]
            user_intent = self.bert.encode(_items_df["genres"].tolist())
            user_intent = cosine_similarity(user_intent, user_intent)[
                np.triu_indices(user_intent.shape[0], k=1)
            ]
            user_intent = (user_intent + 1) / 2
            user_intent = user_intent.mean() * (1 - user_intent.std())

        elif self.user_intent == "item_genre_date_emb":
            _items_metadata = self.items_metadata[
                self.items_metadata.index.isin(list(self.correctly_recommended))
            ]
            _items_df = self.items_df[
                self.items_df.item_id.isin(list(self.correctly_recommended))
            ]
            _items_df = pd.concat([_items_df, _items_metadata], axis=1)[
                ["release_date", "genres"]
            ]
            _items_df["input"] = _items_df.apply(
                lambda x: f"{x['release_date']}, {x['genres']}", axis=1
            )
            user_intent = self.bert.encode(_items_df["input"].tolist())
            user_intent = cosine_similarity(user_intent, user_intent)[
                np.triu_indices(user_intent.shape[0], k=1)
            ]
            user_intent = (user_intent + 1) / 2
            user_intent = user_intent.mean() * (1 - user_intent.std())

        elif self.user_intent == "item_title_emb":
            _items_df = self.items_df[
                self.items_df["item_id"].isin(list(self.correctly_recommended))
            ]

            user_intent = self.bert.encode(_items_df["title"].tolist())
            user_intent = cosine_similarity(user_intent, user_intent)[
                np.triu_indices(user_intent.shape[0], k=1)
            ]
            user_intent = (user_intent + 1) / 2
            user_intent = user_intent.mean() * (1 - user_intent.std())

        elif self.user_intent == "item_title_genre_emb":
            _items_df = self.items_df[
                self.items_df["item_id"].isin(list(self.correctly_recommended))
            ]

            user_intent = self.items_metadata[
                self.items_metadata.index.isin(list(self.correctly_recommended))
            ]
            user_intent = user_intent.drop(["genres"], axis=1).values

            sentence_embeddings = self.bert.encode(_items_df["title"].tolist())

            user_intent = np.concatenate((user_intent, sentence_embeddings), axis=1)

            user_intent = cosine_similarity(user_intent, user_intent)[
                np.triu_indices(user_intent.shape[0], k=1)
            ]
            user_intent = (user_intent + 1) / 2
            user_intent = user_intent.mean() * (1 - user_intent.std())

        else:
            raise "Not valid user intent"
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

                if reward > 0:
                    fair_reward = (
                        fair_reward + 1
                        if user_intent <= self.user_intent_threshold
                        else reward
                    )
                else:
                    fair_reward = (
                        fair_reward
                        if user_intent <= self.user_intent_threshold
                        else reward
                    )

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
                _fair_reward = self.get_fair_reward(group, _reward)
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

            _reward = self.get_reward(action)
            _fair_reward = self.get_fair_reward(group, _reward)
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

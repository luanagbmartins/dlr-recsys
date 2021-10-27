import os
import math
from tqdm import tqdm
import json
import pickle

import obp
from obp_dataset import MovieLensDataset
from sklearn.linear_model import LogisticRegression

import numpy as np

from src.model.simulator import run_bandit_simulation
from src.model.bandits import EpsilonGreedy, LinUCB, WFairLinUCB, FairLinUCB

import wandb

TRAINER = dict(
    epsilon_greedy=EpsilonGreedy,
    linucb=LinUCB,
    wfair_linucb=WFairLinUCB,
    fair_linucb=FairLinUCB,
)


class ContextualBandit:
    def __init__(
        self,
        env,
        users_num,
        items_num,
        genres_num,
        state_size,
        model_path,
        embedding_network_weights_path,
        emb_model,
        train_version,
        data_path,
        movies_groups,
        epsilon,
        epochs,
        embedding_dim=50,
        n_groups=4,
        fairness_constraints=[0.25, 0.25, 0.25, 0.25],
        no_cuda=False,
    ):

        self.dataset = MovieLensDataset(
            data_path=data_path,
            embedding_network_weights_path=embedding_network_weights_path,
            embedding_dim=50,
            users_num=users_num,
            items_num=items_num,
            state_size=state_size,
        )
        self.bandit_feedback = self.dataset.obtain_batch_bandit_feedback()

        self.epsilon = epsilon
        self.n_groups = n_groups
        self.movies_groups = movies_groups
        self.fairness_constraints = fairness_constraints

    def train(self, max_episode_num, top_k=False, load_model=False):
        bandit = TRAINER[self.train_version](
            n_actions=self.dataset.n_actions,
            epsilon=self.epsilon,
            n_group=self.n_groups,
            item_group=self.movies_groups,
            fairness_weight=self.fairness_constraints,
        )

        (
            eg_action_dist,
            eg_aligned_cvr,
            eg_cvr,
            eg_propfair,
            eg_ufg,
            eg_group_count,
        ) = run_bandit_simulation(
            bandit_feedback=self.bandit_feedback,
            policy=bandit,
            epochs=self.epochs,
        )

        # estimate the policy value of the online bandit algorithms using RM
        ope = OffPolicyEvaluation(
            bandit_feedback=self.bandit_feedback,
            ope_estimators=[
                ReplayMethod(),
                DoublyRobust(estimator_name="DR"),
                InverseProbabilityWeighting(estimator_name="IPW"),
                SNIPS(estimator_name="SNIPS"),
                DirectMethod(estimator_name="DM"),
            ],
        )

        # obp.ope.RegressionModel
        regression_model = RegressionModel(
            n_actions=self.dataset.n_actions,  # number of actions; |A|
            len_list=self.dataset.len_list,  # number of items in a recommendation list; K
            base_model=self.LogisticRegression(C=100, max_iter=100000),
        )

        estimated_rewards = regression_model.fit_predict(
            context=self.bandit_feedback["context"],
            action=self.bandit_feedback["action"],
            reward=self.bandit_feedback["reward"],
            position=self.bandit_feedback["position"],
        )

    def update_model(self):
        pass

    def evaluate(self, env, top_k=0, available_items=None):
        # episodic reward
        episode_reward = 0
        correct_count = 0
        steps = 0

        mean_precision = 0
        mean_ndcg = 0

        # Environment
        user_id, items_ids, done = env.reset()

        while not done:
            # Observe current state and Find action
            ## Embedding
            user_eb = self.user_embeddings[user_id]
            items_eb = self.get_items_emb(items_ids)

            with torch.no_grad():
                ## SRM state
                state = self.srm_ave(
                    [
                        user_eb.unsqueeze(0),
                        items_eb.unsqueeze(0),
                    ]
                )

                ## Action(ranking score)
                action = self.actor.network(state)

            ## Item
            recommended_item = self.recommend_item(
                action,
                env.recommended_items,
                top_k=top_k,
                items_ids=list(available_items),
            )

            # Calculate reward and observe new state (in env)
            ## Step
            next_items_ids, reward, done, _ = env.step(recommended_item, top_k=top_k)
            if top_k:
                correct_list = [1 if r > 0 else 0 for r in reward]
                # ndcg
                dcg, idcg = self.calculate_ndcg(
                    correct_list, [1 for _ in range(len(reward))]
                )
                mean_ndcg += dcg / idcg

                # precision
                correct_num = top_k - correct_list.count(0)
                mean_precision += correct_num / top_k
            else:
                mean_precision = reward

            reward = np.sum(reward)
            items_ids = next_items_ids
            episode_reward += reward
            steps += 1
            available_items = (
                available_items - set(recommended_item) if available_items else None
            )

        mean_precision = mean_precision / steps
        mean_ndcg = mean_ndcg / steps

        propfair = 0
        total_exp = np.sum(list(env.group_count.values()))
        if total_exp > 0:
            propfair = np.sum(
                np.array(self.fairness_constraints)
                * np.log(1 + np.array(list(env.group_count.values())) / total_exp)
            )

        return (mean_precision, mean_ndcg, propfair)

    def calculate_ndcg(self, rel, irel):
        dcg = 0
        idcg = 0
        rel = [1 if r > 0 else 0 for r in rel]
        for i, (r, ir) in enumerate(zip(rel, irel)):
            dcg += (r) / np.log2(i + 2)
            idcg += (ir) / np.log2(i + 2)

        return dcg, idcg

    def save_model(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load_model(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

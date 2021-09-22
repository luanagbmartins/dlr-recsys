import os
import math
from tqdm import tqdm
from src.model.replay_buffer import PriorityExperienceReplay
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_math_ops import Exp

from src.model.actor import Actor
from src.model.critic import Critic
from src.model.replay_memory import ReplayMemory
from src.model.embedding import MovieGenreEmbedding, UserMovieEmbedding
from src.model.state_representation import FairRecStateRepresentation

import matplotlib.pyplot as plt

import wandb


class FairRecAgent:
    def __init__(
        self,
        env,
        users_num,
        items_num,
        state_size,
        srm_size,
        model_path,
        embedding_network_weights_path,
        train_version,
        is_test=False,
        use_wandb=False,
        embedding_dim=100,
        actor_hidden_dim=128,
        actor_learning_rate=0.001,
        critic_hidden_dim=128,
        critic_learning_rate=0.001,
        discount_factor=0.9,
        tau=0.001,
        replay_memory_size=1000000,
        learning_starts=1000,
        batch_size=32,
        n_groups=4,
        fairness_constraints=[0.25, 0.25, 0.25, 0.25],
    ):

        self.env = env

        self.users_num = users_num
        self.items_num = items_num

        self.model_path = model_path
        self.embedding_network_weights_path = embedding_network_weights_path

        self.embedding_dim = embedding_dim
        self.srm_size = srm_size
        self.actor_hidden_dim = actor_hidden_dim
        self.actor_learning_rate = actor_learning_rate
        self.critic_hidden_dim = critic_hidden_dim
        self.critic_learning_rate = critic_learning_rate
        self.discount_factor = discount_factor
        self.tau = tau

        self.learning_starts = learning_starts
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size

        self.n_groups = n_groups
        self.fairness_constraints = fairness_constraints

        self.actor = Actor(
            self.embedding_dim,
            self.srm_size,
            self.actor_hidden_dim,
            self.actor_learning_rate,
            state_size,
            self.tau,
        )
        self.critic = Critic(
            self.critic_hidden_dim,
            self.critic_learning_rate,
            self.embedding_dim,
            self.srm_size,
            self.tau,
        )

        self.embedding_network = UserMovieEmbedding(
            users_num, items_num, self.embedding_dim
        )
        self.embedding_network([np.zeros((1,)), np.zeros((1,))])
        self.embedding_network.load_weights(
            "model/movie_lens_1m_fair/movie_lens_1m_fair_2021-09-16_13-46-33/user_movie.h5"
        )  # self.embedding_network_weights_path)

        self.srm_ave = FairRecStateRepresentation(self.embedding_dim, self.n_groups)
        self.srm_ave(
            [
                np.zeros(
                    (
                        1,
                        state_size,
                        self.embedding_dim,
                    )
                ),
                np.zeros((1, 1, self.embedding_dim)),
                np.zeros((1, self.n_groups)),
            ]
        )

        self.buffer = PriorityExperienceReplay(
            self.replay_memory_size,
            self.embedding_dim,
            self.srm_size * self.embedding_dim,
        )
        self.epsilon_for_priority = 1e-6

        self.epsilon = 1.0
        self.epsilon_decay = (self.epsilon - 0.1) / 500000
        self.std = 1.5

        self.is_test = is_test

        # wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=train_version,
                config={
                    "users_num": users_num,
                    "items_num": items_num,
                    "state_size": state_size,
                    "embedding_dim": self.embedding_dim,
                    "actor_hidden_dim": self.actor_hidden_dim,
                    "actor_learning_rate": self.actor_learning_rate,
                    "critic_hidden_dim": self.critic_hidden_dim,
                    "critic_learning_rate": self.critic_learning_rate,
                    "discount_factor": self.discount_factor,
                    "tau": self.tau,
                    "replay_memory_size": self.replay_memory_size,
                    "batch_size": self.batch_size,
                    "std_for_exploration": self.std,
                    "fairness_constraints": self.fairness_constraints,
                },
            )

    def calculate_td_target(self, rewards, q_values, dones):
        y_t = np.copy(q_values)
        for i in range(q_values.shape[0]):
            y_t[i] = rewards[i] + (1 - dones[i]) * (self.discount_factor * q_values[i])
        return y_t

    def recommend_item(self, action, recommended_items, top_k=False, items_ids=None):
        if items_ids == None:
            items_ids = np.array(
                list(set(i for i in range(self.items_num)) - recommended_items)
            )

        items_ebs = self.embedding_network.get_layer("movie_embedding")(items_ids)
        action = tf.transpose(action, perm=(1, 0))
        if top_k:
            item_indice = np.argsort(
                tf.transpose(tf.keras.backend.dot(items_ebs, action), perm=(1, 0))
            )[0][-top_k:]
            return items_ids[item_indice]
        else:
            item_idx = np.argmax(tf.keras.backend.dot(items_ebs, action))
            return items_ids[item_idx]

    def train(self, max_episode_num, top_k=False, load_model=False):
        self.actor.update_target_network()
        self.critic.update_target_network()

        if load_model:
            # Get list of checkpoints
            actor_checkpoints = sorted(
                [
                    int(f.split("_")[1])
                    for f in os.listdir(self.model_path)
                    if f.startswith("actor")
                ]
            )
            critic_checkpoints = sorted(
                [
                    int(f.split("_")[1])
                    for f in os.listdir(self.model_path)
                    if f.startswith("critic")
                ]
            )
            self.load_model(
                os.path.join(
                    self.model_path, "actor_{}.h5".format(actor_checkpoints[-1])
                ),
                os.path.join(
                    self.model_path, "critic_{}.h5".format(critic_checkpoints[-1])
                ),
            )
            print("----- Completely load weights!")

        episodic_precision_history = []

        for episode in tqdm(range(max_episode_num)):
            # episodic reward
            episode_reward = 0
            correct_count = 0
            steps = 0
            q_loss = 0
            mean_action = 0

            # environment
            user_id, items_ids, done = self.env.reset()

            while not done:
                # observe current state & Find action
                # user_eb = self.embedding_network.get_layer("user_embedding")(
                #     np.array(user_id)
                # )
                items_eb = self.embedding_network.get_layer("movie_embedding")(
                    np.array(items_ids)
                )

                groups_eb = []
                for items in items_ids:
                    groups_eb.append(
                        self.embedding_network.get_layer("movie_embedding")(
                            np.array(
                                [
                                    k - 1
                                    for k, v in self.env.movies_groups.items()
                                    if v == self.env.movies_groups[items]
                                ]
                            )
                        )
                    )

                fairness_allocation = []
                for group in range(self.n_groups):
                    _group = group + 1
                    if _group not in self.env.group_count:
                        self.env.group_count[_group] = 0
                    fairness_allocation.append(
                        self.env.group_count[_group] / self.env.total_recommended_items
                        if self.env.total_recommended_items > 0
                        else 0
                    )

                ## SRM state
                state = self.srm_ave(
                    [
                        np.expand_dims(items_eb, axis=0),
                        groups_eb,
                        np.expand_dims(fairness_allocation, axis=0),
                    ]
                )

                ## action(ranking score)
                action = self.actor.network(state)

                ## epsilon-greedy exploration
                if self.epsilon > np.random.uniform() and not self.is_test:
                    self.epsilon -= self.epsilon_decay
                    action += np.random.normal(0, self.std, size=action.shape)

                ## item
                recommended_item = self.recommend_item(
                    action, self.env.recommended_items, top_k=top_k
                )

                # calculate reward and observe new state
                ## Step
                next_items_ids, reward, done, _ = self.env.step(
                    recommended_item, top_k=top_k
                )
                if top_k:
                    reward = np.sum(reward)

                # get next_state
                next_items_eb = self.embedding_network.get_layer("movie_embedding")(
                    np.array(next_items_ids)
                )

                groups_eb = []
                for items in items_ids:
                    groups_eb.append(
                        self.embedding_network.get_layer("movie_embedding")(
                            np.array(
                                [
                                    k - 1
                                    for k, v in self.env.movies_groups.items()
                                    if v == self.env.movies_groups[items]
                                ]
                            )
                        )
                    )

                fairness_allocation = []
                for group in range(self.n_groups):
                    _group = group + 1
                    if _group not in self.env.group_count:
                        self.env.group_count[_group] = 0
                    fairness_allocation.append(
                        self.env.group_count[_group] / self.env.total_recommended_items
                    )

                ## SRM state
                next_state = self.srm_ave(
                    [
                        np.expand_dims(next_items_eb, axis=0),
                        groups_eb,
                        np.expand_dims(fairness_allocation, axis=0),
                    ]
                )

                # buffer
                self.buffer.append(state, action, reward, next_state, done)

                if self.buffer.crt_idx > self.learning_starts or self.buffer.is_full:
                    # sample a minibatch
                    (
                        batch_states,
                        batch_actions,
                        batch_rewards,
                        batch_next_states,
                        batch_dones,
                        weight_batch,
                        index_batch,
                    ) = self.buffer.sample(self.batch_size)

                    # set TD targets
                    target_next_action = self.actor.target_network(batch_next_states)
                    qs = self.critic.network([target_next_action, batch_next_states])
                    target_qs = self.critic.target_network(
                        [target_next_action, batch_next_states]
                    )
                    min_qs = tf.raw_ops.Min(
                        input=tf.concat([target_qs, qs], axis=1), axis=1, keep_dims=True
                    )  # Double Q method
                    td_targets = self.calculate_td_target(
                        batch_rewards, min_qs, batch_dones
                    )

                    # update priority
                    for (p, i) in zip(td_targets, index_batch):
                        self.buffer.update_priority(
                            abs(p[0]) + self.epsilon_for_priority, i
                        )

                    # update critic network
                    q_loss += self.critic.train(
                        [batch_actions, batch_states], td_targets, weight_batch
                    )

                    # update actor network
                    s_grads = self.critic.dq_da([batch_actions, batch_states])
                    self.actor.train(batch_states, s_grads)
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                items_ids = next_items_ids
                episode_reward += reward
                mean_action += np.sum(action[0]) / (len(action[0]))
                steps += 1

                if reward > 0:
                    correct_count += 1

                propfair = 0
                for group in range(self.n_groups):
                    _group = group + 1
                    if _group not in self.env.group_count:
                        self.env.group_count[_group] = 0

                    propfair += self.fairness_constraints[group] * math.log(
                        1
                        + (
                            self.env.group_count[_group]
                            / self.env.total_recommended_items
                        )
                    )
                cvr = correct_count / self.env.total_recommended_items
                wandb.log(
                    {
                        "propfair": propfair,
                        "cvr": cvr,
                        "ufg": propfair / max(1 - cvr, 0.01),
                    }
                )

                print("----------")
                print("- recommended items: ", self.env.total_recommended_items)
                print("- group count: ", self.env.group_count)
                print("- propfair: ", propfair)
                print("- cvr: ", cvr)
                print("- ufg: ", propfair / max(1 - cvr, 0.01))
                print("- epsilon: ", self.epsilon)
                print("- reward: ", reward)
                print()

                if done:
                    precision = int(correct_count / steps * 100)
                    print("----------")
                    print("- precision: ", precision)
                    print("- total_reward: ", episode_reward)
                    print("- q_loss: ", q_loss / steps)
                    print("- mean_action: ", mean_action / steps)
                    print()
                    if self.use_wandb:
                        wandb.log(
                            {
                                "precision": precision,
                                "total_reward": episode_reward,
                                "epsilone": self.epsilon,
                                "q_loss": q_loss / steps,
                                "mean_action": mean_action / steps,
                            }
                        )
                    episodic_precision_history.append(precision)

            if (episode + 1) % 50 == 0:
                plt.plot(episodic_precision_history)
                plt.savefig(
                    os.path.join(
                        self.model_path, "images/training_precision_%_top_5.png"
                    )
                )

            if (episode + 1) % 1000 == 0:
                self.save_model(
                    os.path.join(self.model_path, "actor_{}.h5".format(episode + 1)),
                    os.path.join(self.model_path, "critic_{}.h5".format(episode + 1)),
                )

    def evaluate(self, env, top_k=False):
        # episodic reward
        episode_reward = 0
        correct_count = 0
        steps = 0
        mean_precision = 0
        mean_ndcg = 0
        mean_cvr = 0
        mean_propfair = 0
        mean_ufg = 0
        # Environment
        user_id, items_ids, done = env.reset()

        while not done:
            # Observe current state and Find action
            items_eb = self.embedding_network.get_layer("movie_embedding")(
                np.array(items_ids)
            )

            groups_eb = []
            for items in items_ids:
                groups_eb.append(
                    self.embedding_network.get_layer("movie_embedding")(
                        np.array(
                            [
                                k - 1
                                for k, v in env.movies_groups.items()
                                if v == env.movies_groups[items]
                            ]
                        )
                    )
                )

            fairness_allocation = []
            for group in range(self.n_groups):
                _group = group + 1
                if _group not in env.group_count:
                    env.group_count[_group] = 0
                fairness_allocation.append(
                    env.group_count[_group] / env.total_recommended_items
                    if env.total_recommended_items > 0
                    else 0
                )

            ## SRM state
            state = self.srm_ave(
                [
                    np.expand_dims(items_eb, axis=0),
                    groups_eb,
                    np.expand_dims(fairness_allocation, axis=0),
                ]
            )
            ## Action(ranking score)
            action = self.actor.network(state)

            ## Item
            recommended_item = self.recommend_item(
                action, env.recommended_items, top_k=top_k
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

            reward = np.sum(reward)
            items_ids = next_items_ids
            episode_reward += reward
            steps += 1

            propfair = 0
            for group in range(10):
                _group = group + 1
                if _group not in env.group_count:
                    env.group_count[_group] = 0

                propfair += self.fairness_constraints[group] * math.log(
                    1 + (env.group_count[_group] / len(recommended_item))
                )

            cvr = correct_num / len(recommended_item)
            ufg = propfair / max(1 - cvr, 0.01)

            mean_propfair += propfair
            mean_cvr += cvr
            mean_ufg += ufg

        return (
            mean_precision / steps,
            mean_ndcg / steps,
            mean_propfair / steps,
            mean_cvr / steps,
            mean_ufg / steps,
        )

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

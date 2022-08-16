import os
from tqdm import tqdm

import torch
import numpy as np

import time

from src.model.pmf import PMF
from src.model.actor import Actor
from src.model.critic import Critic
from src.model.ou_noise import OUNoise
from src.model.replay_buffer import PriorityExperienceReplay
from src.model.state_representation import StateRepresentation

import wandb


class DRRAgent:
    def __init__(
        self,
        env,
        users_num,
        items_num,
        state_size,
        srm_size,
        srm_type,
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
        learning_starts=1000,
        replay_memory_size=1000000,
        batch_size=32,
        n_groups=4,
        fairness_constraints=[0.25, 0.25, 0.25, 0.25],
        no_cuda=False,
        use_reward_model=True,
    ):
        # no_cuda = True
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        )

        self.env = env
        self.state_size = state_size
        self.users_num = users_num
        self.items_num = items_num

        self.model_path = model_path

        self.embedding_network_weights_path = embedding_network_weights_path

        self.embedding_dim = embedding_dim
        self.srm_size = srm_size
        self.srm_type = srm_type
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
            self.device,
        )
        self.critic = Critic(
            self.critic_hidden_dim,
            self.critic_learning_rate,
            self.embedding_dim,
            self.srm_size,
            self.tau,
            self.device,
        )

        self.srm = StateRepresentation(
            state_size=state_size,
            embedding_dim=self.embedding_dim,
            n_groups=self.n_groups,
            state_representation_type=self.srm_type,
            learning_rate=0.001,
            device=self.device,
        )

        self.reward_model = PMF(users_num, items_num, self.embedding_dim).to(
            self.device
        )
        self.reward_model.load_state_dict(
            torch.load(
                self.embedding_network_weights_path,
                map_location=torch.device(self.device),
            )
        )

        self.user_embeddings = self.reward_model.user_embeddings.weight.data
        self.item_embeddings = self.reward_model.item_embeddings.weight.data

        print("----- Reward Model: ", use_reward_model)
        if self.env and use_reward_model:
            self.env.reward_model = self.reward_model
            self.env.item_embeddings = self.item_embeddings
            self.env.device = self.device

        self.buffer = PriorityExperienceReplay(
            buffer_size=self.replay_memory_size,
            embedding_dim=self.embedding_dim,
            obs_size=1 + self.state_size + self.n_groups,
            device=self.device,
        )
        self.epsilon_for_priority = 1e-6

        # noise
        self.noise = OUNoise(self.embedding_dim, decay_period=10)

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
                    "learning_starts": self.learning_starts,
                    "replay_memory_size": self.replay_memory_size,
                    "batch_size": self.batch_size,
                    "group_fairness": n_groups,
                    "fairness_constraints": self.fairness_constraints,
                    "reward_model": use_reward_model,
                },
            )

    def calculate_td_target(self, rewards, q_values, dones):
        return rewards + ((1 - dones.long()) * (self.discount_factor * q_values))

    def recommend_item(self, action, recommended_items, top_k=False, items_ids=None):
        if items_ids == None:
            items_ids = list(set(i for i in range(self.items_num)) - recommended_items)

        items_ids = np.array(items_ids)
        with torch.no_grad():
            items_ebs = self.get_items_emb(items_ids)
        action = torch.transpose(action, 1, 0).float()

        if top_k:
            item_indice = torch.argsort(
                torch.transpose(torch.matmul(items_ebs, action), 1, 0)
            )[0][-top_k:]
            return items_ids[item_indice.detach().cpu().numpy()]
        else:
            item_idx = torch.argmax(torch.matmul(items_ebs, action))
            return items_ids[item_idx]

    def get_items_emb(self, items_ids):
        items_eb = self.item_embeddings[items_ids]

        return items_eb

    def get_state(self, user_id, items_ids, group_counts=None):
        items_emb = self.get_items_emb(items_ids)

        ## SRM state
        state = self.srm.network(
            [
                self.user_embeddings[user_id],
                items_emb.unsqueeze(0) if len(items_emb.shape) < 3 else items_emb,
            ]
        )

        return state

    def train(self, max_episode_num, top_k=False, load_model=False):
        self.actor.update_target_network()
        self.critic.update_target_network()

        if load_model:
            # Get list of checkpoints
            actor_checkpoint = sorted(
                [
                    int((f.split("_")[1]).split(".")[0])
                    for f in os.listdir(self.model_path)
                    if f.startswith("actor_")
                ]
            )[-1]
            critic_checkpoint = sorted(
                [
                    int((f.split("_")[1]).split(".")[0])
                    for f in os.listdir(self.model_path)
                    if f.startswith("critic_")
                ]
            )[-1]
            srm_checkpoint = sorted(
                [
                    int((f.split("_")[1]).split(".")[0])
                    for f in os.listdir(self.model_path)
                    if f.startswith("srm_")
                ]
            )[-1]

            self.load_model(
                os.path.join(self.model_path, "actor_{}.h5".format(actor_checkpoint)),
                os.path.join(self.model_path, "critic_{}.h5".format(critic_checkpoint)),
                os.path.join(self.model_path, "srm_{}.h5".format(srm_checkpoint)),
            )

        sum_precision = 0
        sum_ndcg = 0
        sum_propfair = 0
        sum_reward = 0

        for episode in tqdm(range(max_episode_num)):

            # episodic reward
            episode_reward = 0
            steps = 0
            critic_loss = 0
            actor_loss = 0
            mean_action = 0
            mean_precision = 0
            mean_ndcg = 0

            list_recommended_item = []

            # environment
            user_id, items_ids, done = self.env.reset()
            self.noise.reset()

            while not done:
                with torch.no_grad():
                    # observe current state & Find action
                    group_counts = self.env.get_group_count()
                    state = self.get_state(
                        np.array([user_id]),
                        np.array([items_ids]),
                        np.array([group_counts]),
                    )

                    ## action(ranking score)
                    action = self.actor.network(state.detach())

                ## ou exploration
                if not self.is_test:
                    action = self.noise.get_action(
                        action.detach().cpu().numpy()[0], steps
                    ).to(self.device)

                ## item
                recommended_item = self.recommend_item(
                    action, self.env.get_recommended_items(), top_k=top_k
                )
                list_recommended_item.append(recommended_item)

                # calculate reward and observe new state
                ## Step
                next_items_ids, reward, done, info = self.env.step(
                    recommended_item, top_k=top_k
                )

                # get next_state
                # next_state = self.get_state([user_id], [[next_items_ids]])
                next_group_counts = self.env.get_group_count()

                # experience replay
                self.buffer.append(
                    torch.Tensor(
                        np.concatenate(([user_id], items_ids, group_counts), axis=0)
                    ).to(self.device),
                    action,
                    torch.FloatTensor([np.sum(reward)]).to(self.device)
                    if top_k
                    else torch.FloatTensor([reward]).to(self.device),
                    torch.Tensor(
                        np.concatenate(
                            ([user_id], next_items_ids, next_group_counts), axis=0
                        )
                    ).to(self.device),
                    torch.Tensor([done]).to(self.device),
                )

                if self.buffer.crt_idx > self.learning_starts or self.buffer.is_full:
                    _critic_loss, _actor_loss = self.update_model()
                    actor_loss += _actor_loss
                    critic_loss += _critic_loss

                items_ids = next_items_ids
                episode_reward += np.sum(reward) if top_k else reward

                mean_action += np.sum(action[0].cpu().numpy()) / (
                    len(action[0].cpu().numpy())
                )
                steps += 1

                if top_k:
                    correct_list = info["precision"]
                    # ndcg
                    dcg, idcg = self.calculate_ndcg(
                        correct_list, [1 for _ in range(len(info["precision"]))]
                    )
                    mean_ndcg += dcg / idcg

                    # precision
                    correct_num = top_k - correct_list.count(0)
                    mean_precision += correct_num / top_k
                else:
                    mean_precision += info["precision"]

                if done:
                    propfair = 0
                    total_exp = np.sum(self.env.get_group_count())
                    if total_exp > 0:
                        propfair = np.sum(
                            np.array(self.fairness_constraints)
                            * np.log(
                                1 + np.array(self.env.get_group_count()) / total_exp
                            )
                        )

                    sum_precision += mean_precision / steps
                    sum_ndcg += mean_ndcg / steps
                    sum_propfair += propfair
                    sum_reward += episode_reward

                    if self.use_wandb:
                        wandb.log(
                            {
                                "precision": (mean_precision / steps) * 100,
                                "ndcg": mean_ndcg / steps,
                                "total_reward": episode_reward,
                                "critic_loss": critic_loss / steps,
                                "actor_loss": actor_loss / steps,
                                "mean_action": mean_action / steps,
                                "propfair": propfair,
                                "cvr": mean_precision / steps,
                                "ufg": propfair
                                / max(1 - (mean_precision / steps), 0.01),
                            }
                        )

            if (episode + 1) % 1000 == 0:
                self.save_model(
                    os.path.join(self.model_path, "actor_{}.h5".format(episode + 1)),
                    os.path.join(self.model_path, "critic_{}.h5".format(episode + 1)),
                    os.path.join(self.model_path, "srm_{}.h5".format(episode + 1)),
                )

        return (
            sum_precision / max_episode_num,
            sum_ndcg / max_episode_num,
            sum_propfair / max_episode_num,
            sum_reward / max_episode_num,
            {user_id: list_recommended_item},
        )

    def update_model(self):
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

        states = self.get_state(
            batch_states[:, 0].long().detach().cpu().numpy(),  # user_id
            batch_states[:, 1 : self.state_size + 1]
            .long()
            .detach()
            .cpu()
            .numpy(),  # items_ids
            batch_states[:, self.state_size + 1 :]
            .long()
            .detach()
            .cpu()
            .numpy(),  # group_counts
        )
        actions = self.actor.network(states)

        policy_loss = self.critic.network(
            [
                actions,
                states,
            ]
        )
        policy_loss = -policy_loss.mean()

        # Estimate target Q value
        next_states = self.get_state(
            batch_next_states[:, 0].long().detach().cpu().numpy(),
            batch_next_states[:, 1 : self.state_size + 1].long().detach().cpu().numpy(),
            batch_next_states[:, self.state_size + 1 :].long().detach().cpu().numpy(),
        )
        target_next_action = self.actor.target_network(next_states)
        target_qs = self.critic.target_network(
            [
                target_next_action.detach(),
                next_states,
            ]
        )
        qs = self.critic.network(
            [
                target_next_action.detach(),
                next_states,
            ]
        )

        min_qs = torch.min(torch.cat([target_qs, qs], axis=1), 1, True).values.squeeze(
            1
        )  # Double Q method

        # set TD targets
        td_targets = self.calculate_td_target(
            batch_rewards,
            min_qs,
            batch_dones,
        ).unsqueeze(1)

        # update priority
        for (p, i) in zip(td_targets, index_batch):
            self.buffer.update_priority(abs(p[0]) + self.epsilon_for_priority, i)

        # get Q values for current state
        value = self.critic.network(
            [
                batch_actions,
                states,
            ]
        )
        value_loss = self.critic.loss(value, td_targets)
        value_loss = torch.mean(value_loss * weight_batch)

        # update actor network
        self.srm.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # update critic network
        self.critic.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic.optimizer.step()
        self.srm.optimizer.step()

        # target update
        self.actor.update_target_network()
        self.critic.update_target_network()

        return value_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()

    def online_evaluate(self, env, top_k=0, available_items=None, load_model=False):

        if load_model:
            # Get list of checkpoints
            actor_checkpoint = sorted(
                [
                    int((f.split("_")[1]).split(".")[0])
                    for f in os.listdir(self.model_path)
                    if f.startswith("actor_")
                ]
            )[-1]
            critic_checkpoint = sorted(
                [
                    int((f.split("_")[1]).split(".")[0])
                    for f in os.listdir(self.model_path)
                    if f.startswith("critic_")
                ]
            )[-1]
            srm_checkpoint = sorted(
                [
                    int((f.split("_")[1]).split(".")[0])
                    for f in os.listdir(self.model_path)
                    if f.startswith("srm_")
                ]
            )[-1]

            self.load_model(
                os.path.join(self.model_path, "actor_{}.h5".format(actor_checkpoint)),
                os.path.join(self.model_path, "critic_{}.h5".format(critic_checkpoint)),
                os.path.join(self.model_path, "srm_{}.h5".format(srm_checkpoint)),
            )

        steps = 0
        mean_precision = 0
        mean_ndcg = 0
        episode_reward = 0

        critic_loss = 0
        actor_loss = 0

        list_recommended_item = []

        # Environment
        user_id, items_ids, done = env.reset()
        self.noise.reset()

        while not done:
            with torch.no_grad():
                # observe current state & Find action
                group_counts = env.get_group_count()
                state = self.get_state(
                    np.array([user_id]),
                    np.array([items_ids]),
                    np.array([group_counts]),
                )

                ## action(ranking score)
                action = self.actor.network(state.detach())

            ## ou exploration
            if not self.is_test:
                action = self.noise.get_action(
                    action.detach().cpu().numpy()[0], steps
                ).to(self.device)

            ## Item
            recommended_item = self.recommend_item(
                action,
                env.get_recommended_items(),
                top_k=top_k,
                items_ids=list(available_items) if available_items else None,
            )
            list_recommended_item.extend(
                list([recommended_item] if not top_k else recommended_item)
            )

            # Calculate reward and observe new state (in env)
            ## Step
            next_items_ids, reward, done, info = env.step(recommended_item, top_k=top_k)

            # get next_state
            # next_state = self.get_state([user_id], [[next_items_ids]])
            next_group_counts = env.get_group_count()

            # experience replay
            self.buffer.append(
                torch.Tensor(
                    np.concatenate(([user_id], items_ids, group_counts), axis=0)
                ).to(self.device),
                action,
                torch.FloatTensor([np.sum(reward)]).to(self.device)
                if top_k
                else torch.FloatTensor([reward]).to(self.device),
                torch.Tensor(
                    np.concatenate(
                        ([user_id], next_items_ids, next_group_counts), axis=0
                    )
                ).to(self.device),
                torch.Tensor([done]).to(self.device),
            )

            if self.buffer.crt_idx > self.learning_starts or self.buffer.is_full:
                _critic_loss, _actor_loss = self.update_model()
                actor_loss += _actor_loss
                critic_loss += _critic_loss

            items_ids = next_items_ids
            episode_reward += np.sum(reward) if top_k else reward
            steps += 1

            if top_k:
                correct_list = info["precision"]
                # ndcg
                dcg, idcg = self.calculate_ndcg(
                    correct_list, [1 for _ in range(len(info["precision"]))]
                )
                mean_ndcg += dcg / idcg

                # precision
                correct_num = top_k - correct_list.count(0)
                mean_precision += correct_num / top_k
            else:
                mean_precision += info["precision"]

            available_items = (
                available_items - set(recommended_item) if available_items else None
            )

        propfair = 0
        total_exp = np.sum(env.get_group_count())
        if total_exp > 0:
            propfair = np.sum(
                np.array(self.fairness_constraints)
                * np.log(1 + np.array(env.get_group_count()) / total_exp)
            )

        return {
            "precision": mean_precision / steps,
            "propfair": propfair,
            "reward": episode_reward,
            "recommended_items": {user_id: list_recommended_item},
            "exposure": (np.array(env.get_group_count()) / total_exp).tolist(),
            "critic_loss": critic_loss / steps,
            "actor_loss": actor_loss / steps,
        }

    def offline_evaluate(self, env, top_k=0, available_items=None):

        steps = 0
        mean_precision = 0
        mean_ndcg = 0
        episode_reward = 0

        # Environment
        user_id, items_ids, done = env.reset()

        while not done:
            with torch.no_grad():
                with torch.no_grad():
                    # observe current state & Find action
                    group_counts = env.get_group_count()
                    state = self.get_state(
                        np.array([user_id]),
                        np.array([items_ids]),
                        np.array([group_counts]),
                    )

                    ## action(ranking score)
                    action = self.actor.network(state.detach())

            ## Item
            recommended_item = self.recommend_item(
                action,
                env.get_recommended_items(),
                top_k=top_k,
                items_ids=list(available_items),
            )

            # Calculate reward and observe new state (in env)
            ## Step
            next_items_ids, reward, done, info = env.step(recommended_item, top_k=top_k)

            if top_k:
                correct_list = info["precision"]
                # ndcg
                dcg, idcg = self.calculate_ndcg(
                    correct_list, [1 for _ in range(len(info["precision"]))]
                )
                mean_ndcg += dcg / idcg

                # precision
                correct_num = top_k - correct_list.count(0)
                mean_precision += correct_num / top_k

            else:
                mean_precision += info["precision"]

            items_ids = next_items_ids

            steps += 1
            episode_reward += np.sum(reward)

            available_items = (
                available_items - set(recommended_item) if available_items else None
            )

        return (
            mean_precision / steps,
            mean_ndcg / steps,
            episode_reward,
            episode_reward / steps,
        )

    def calculate_ndcg(self, rel, irel):
        dcg = 0
        idcg = 0
        rel = [1 if r > 0 else 0 for r in rel]
        for i, (r, ir) in enumerate(zip(rel, irel)):
            dcg += (r) / np.log2(i + 2)
            idcg += (ir) / np.log2(i + 2)

        return dcg, idcg

    def save_model(self, actor_path, critic_path, srm_path, buffer_path=None):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        self.srm.save_weights(srm_path)
        if buffer_path:
            import pickle

            with open(buffer_path, "wb") as f:
                pickle.dump(self.buffer, f)

    def load_model(self, actor_path, critic_path, srm_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        self.srm.load_weights(srm_path)

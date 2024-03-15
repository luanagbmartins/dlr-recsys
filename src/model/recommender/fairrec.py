from tokenize import group
import torch
import numpy as np
import time
import os

from operator import itemgetter

from src.model.recommender.drr import DRRAgent


class FairRecAgent(DRRAgent):
    def __init__(
        self,
        env,
        users_num,
        items_num,
        state_size,
        srm_size,
        srm_type,
        model_path,
        reward_model_path,
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
        super().__init__(
            env=env,
            users_num=users_num,
            items_num=items_num,
            state_size=state_size,
            srm_size=srm_size,
            srm_type=srm_type,
            model_path=model_path,
            reward_model_path=reward_model_path,
            embedding_network_weights_path=embedding_network_weights_path,
            train_version=train_version,
            is_test=False,
            use_wandb=use_wandb,
            embedding_dim=embedding_dim,
            actor_hidden_dim=actor_hidden_dim,
            actor_learning_rate=actor_learning_rate,
            critic_hidden_dim=critic_hidden_dim,
            critic_learning_rate=critic_learning_rate,
            discount_factor=discount_factor,
            tau=tau,
            learning_starts=learning_starts,
            replay_memory_size=replay_memory_size,
            batch_size=batch_size,
            n_groups=n_groups,
            fairness_constraints=fairness_constraints,
            no_cuda=no_cuda,
            use_reward_model=use_reward_model,
        )

        groups_id = list(self.env.groups_items.keys())
        groups_items = list(self.env.groups_items.values())
        self.group_emb = {}
        for g, items in zip(groups_id, groups_items):
            self.group_emb[g] = torch.mean(self.get_items_emb(items), axis=0)

    def get_state(self, user_id, items_ids, group_counts):
        groups = []
        fairness_allocation = []
        for batch_item, batch_group in zip(items_ids, group_counts):
            _groups = list(itemgetter(*batch_item)(self.env.item_groups))
            groups.append(torch.stack([self.group_emb[g] for g in _groups]))

            total_exp = np.sum(batch_group)
            _fairness_allocation = (
                (np.array(batch_group) / total_exp)
                if total_exp > 0
                else np.zeros(self.n_groups)
            )
            fairness_allocation.append(_fairness_allocation)

        # context_emb = (
        #     torch.from_numpy(self.context_emb.get_embedding(items_ids.tolist())).to(
        #         self.device
        #     )
        #     if self.context_emb
        #     else None
        # )
        context_emb = None

        ## SRM state
        state = self.srm.network(
            [
                self.get_items_emb(items_ids),  # batch_size x n_items x embedding_dim
                torch.stack(groups).to(
                    self.device
                ),  # batch_size x n_items x embedding_dim
                torch.FloatTensor(np.array(fairness_allocation)).to(
                    self.device
                ),  # batch_size x n_groups
                self.user_embeddings[user_id],
                context_emb,
            ]
        )

        return state

    def online_evaluate(self, env, top_k=False, available_items=None, load_model=False):
        env.item_embeddings = self.item_embeddings
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
        buffer_states = []
        buffer_intent = []
        buffer_actions = []
        buffer_propfair = []
        buffer_precision = []
        buffer_reward = []
        buffer_relevance = []
        buffer_fairness = []

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
            # if not self.is_test:
            # action = self.noise.get_action(action.detach().cpu().numpy()[0], steps).to(
            #     self.device
            # )

            buffer_states.append(state.detach().cpu().numpy().tolist())
            buffer_intent.append(env._get_user_intent().tolist())
            buffer_actions.append(action.detach().cpu().numpy().tolist()[0])

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

            propfair = 0
            total_exp = np.sum(env.get_group_count())
            if total_exp > 0:
                propfair = np.sum(
                    np.array(self.fairness_constraints)
                    * np.log(1 + np.array(env.get_group_count()) / total_exp)
                )

            buffer_propfair.append(propfair)

            # get next_state
            # next_state = self.get_state([user_id], [[next_items_ids]])
            next_group_counts = env.get_group_count()

            # experience replay
            self.buffer.append(
                torch.Tensor(
                    np.concatenate(([user_id], items_ids, group_counts), axis=0)
                ).to(self.device),
                action,
                (
                    torch.FloatTensor([np.sum(reward)]).to(self.device)
                    if top_k
                    else torch.FloatTensor([reward]).to(self.device)
                ),
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

            buffer_relevance.append(info["relevance"])
            buffer_fairness.append(info["fairness"])
            buffer_precision.append(info["precision"])
            buffer_reward.append(np.sum(reward) if top_k else reward)

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
            "precision_list": buffer_precision,
            "propfair": propfair,
            "reward": episode_reward,
            "reward_list": buffer_reward,
            "recommended_items": {user_id: list_recommended_item},
            "exposure": (np.array(env.get_group_count()) / total_exp).tolist(),
            "critic_loss": critic_loss / steps,
            "actor_loss": actor_loss / steps,
            "user_id": user_id,
            "user_states": buffer_states,
            "user_intent": buffer_intent,
            "user_action_rank": buffer_actions,
            "user_propfair": buffer_propfair,
            "relevance": buffer_relevance,
            "fairness": buffer_fairness,
        }

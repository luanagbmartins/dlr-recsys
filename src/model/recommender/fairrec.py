import torch
import numpy as np

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

        super().__init__(
            env=env,
            users_num=users_num,
            items_num=items_num,
            state_size=state_size,
            srm_size=srm_size,
            model_path=model_path,
            embedding_network_weights_path=embedding_network_weights_path,
            train_version=train_version,
            is_test=is_test,
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

    def get_state(self, user_id, items_ids, group_counts):

        groups = []
        fairness_allocation = []
        for batch_item, batch_group in zip(items_ids, group_counts):

            _groups = list(itemgetter(*batch_item)(self.env.item_groups))
            groups_id = list(itemgetter(*_groups)(self.env.groups_movies))
            groups.append(
                torch.stack(
                    [torch.mean(self.get_items_emb(g), axis=0) for g in groups_id]
                )
            )

            total_exp = np.sum(batch_group)
            _fairness_allocation = (
                (np.array(batch_group) / total_exp)
                if total_exp > 0
                else np.zeros(self.n_groups)
            )
            fairness_allocation.append(_fairness_allocation)

        ## SRM state
        state = self.srm.network(
            [
                self.get_items_emb(items_ids),  # batch_size x n_items x embedding_dim
                torch.stack(groups).to(
                    self.device
                ),  # batch_size x n_items x embedding_dim
                torch.FloatTensor(fairness_allocation).to(
                    self.device
                ),  # batch_size x n_groups
                self.user_embeddings[user_id],
            ]
        )
        return state

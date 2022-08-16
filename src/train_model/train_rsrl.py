import os
import json
import pickle
import luigi

import torch
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import recmetrics as rm
from random import sample

import plotly.graph_objects as go
import plotly_express as px

from src.environment import OfflineEnv, OfflineFairEnv
from src.model.recommender import DRRAgent, FairRecAgent

AGENT = dict(drr=DRRAgent, fairrec=FairRecAgent)
ENV = dict(drr=OfflineEnv, fairrec=OfflineFairEnv)


class RSRL(luigi.Task):
    """Recommender System with Reinforcement Learning"""

    output_path = luigi.Parameter()

    algorithm: str = luigi.ChoiceParameter(choices=AGENT.keys())
    epochs: int = luigi.IntParameter(default=5)
    users_num: int = luigi.IntParameter(default=6041)
    items_num: int = luigi.IntParameter(default=3953)
    state_size: int = luigi.IntParameter(default=10)
    srm_size: int = luigi.IntParameter(default=3)
    max_eps_num: int = luigi.IntParameter(default=8000)
    embedding_dim: int = luigi.IntParameter(default=100)
    actor_hidden_dim: int = luigi.IntParameter(default=128)
    actor_learning_rate: float = luigi.FloatParameter(default=0.001)
    critic_hidden_dim: int = luigi.IntParameter(default=128)
    critic_learning_rate: float = luigi.FloatParameter(default=0.001)
    discount_factor: float = luigi.FloatParameter(default=0.9)
    tau: float = luigi.FloatParameter(default=0.001)
    learning_starts: int = luigi.IntParameter(default=1000)
    replay_memory_size: int = luigi.IntParameter(default=1000000)
    batch_size: int = luigi.IntParameter(default=32)
    n_groups: int = luigi.IntParameter(default=4)
    fairness_constraints: list = luigi.ListParameter(default=[0.25, 0.25, 0.25, 0.25])
    reward_threshold: float = luigi.FloatParameter(default=4.0)
    reward_version: str = luigi.Parameter(default="paper")
    user_intent_threshold: float = luigi.FloatParameter()
    user_intent: str = luigi.Parameter(default="item_emb_pmf")
    top_k: int = luigi.IntParameter(default=10)
    done_count: int = luigi.IntParameter(default=10)

    embedding_network_weights: str = luigi.Parameter(default="")
    use_reward_model: bool = luigi.BoolParameter(default=True)

    train_version: str = luigi.Parameter()
    use_wandb: bool = luigi.BoolParameter()
    load_model: bool = luigi.BoolParameter()
    dataset_path: str = luigi.Parameter()
    only_evaluate: bool = luigi.BoolParameter(default=False)
    no_cuda: bool = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        super(RSRL, self).__init__(*args, **kwargs)

        self.train_version = "{}_{}".format(self.train_version, self.reward_version)

    def output(self):
        return {
            "actor_model": luigi.LocalTarget(
                os.path.join(self.output_path, "actor.h5")
            ),
            "critic_model": luigi.LocalTarget(
                os.path.join(self.output_path, "critic.h5")
            ),
            "srm_model": luigi.LocalTarget(os.path.join(self.output_path, "srm.h5")),
            "metrics": luigi.LocalTarget(
                os.path.join(self.output_path, "metrics.json")
            ),
        }

    def run(self):
        dataset = self.load_data()

        if not self.only_evaluate:
            self.train_model(dataset)
        self.evaluate_model(dataset)

    def load_data(self):
        with open(self.dataset_path) as json_file:
            _dataset_path = json.load(json_file)

        dataset = {}
        with open(_dataset_path["train_users_dict"], "rb") as pkl_file:
            dataset["train_users_dict"] = pickle.load(pkl_file)

        with open(_dataset_path["eval_users_dict"], "rb") as pkl_file:
            dataset["eval_users_dict"] = pickle.load(pkl_file)

        with open(_dataset_path["item_groups"], "rb") as pkl_file:
            dataset["item_groups"] = pickle.load(pkl_file)

        dataset["items_df"] = pd.read_csv(_dataset_path["items_df"])
        dataset["items_metadata"] = pd.read_csv(_dataset_path["items_metadata"])
        dataset["title_emb"] = _dataset_path["title_emb"]

        return dataset

    def seed_all(self, seed):
        cuda = torch.device(
            "cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu"
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.manual_seed(seed=seed)

    def train_model(self, dataset):
        self.seed_all(0)
        print("---------- Seeds initialized")

        print("---------- Prepare Env")
        print("---------- Algorithm ", self.algorithm)
        print("---------- User Intent ", self.user_intent, self.user_intent_threshold)

        env = ENV[self.algorithm](
            users_dict=dataset["train_users_dict"],
            n_groups=self.n_groups,
            item_groups=dataset["item_groups"],
            items_metadata=dataset["items_metadata"],
            items_df=dataset["items_df"],
            state_size=self.state_size,
            done_count=self.done_count,
            fairness_constraints=self.fairness_constraints,
            reward_threshold=self.reward_threshold,
            reward_version=self.reward_version,
            user_intent_threshold=self.user_intent_threshold,
            user_intent=self.user_intent,
            title_emb_path=dataset["title_emb"],
        )

        print("---------- Initialize Agent")
        recommender = AGENT[self.algorithm](
            env=env,
            users_num=self.users_num,
            items_num=self.items_num,
            srm_size=self.srm_size,
            srm_type="{}_{}".format(self.algorithm, self.reward_version),
            state_size=self.state_size,
            train_version=self.train_version,
            use_wandb=self.use_wandb,
            embedding_dim=self.embedding_dim,
            actor_hidden_dim=self.actor_hidden_dim,
            actor_learning_rate=self.actor_learning_rate,
            critic_hidden_dim=self.critic_hidden_dim,
            critic_learning_rate=self.critic_learning_rate,
            discount_factor=self.discount_factor,
            tau=self.tau,
            learning_starts=self.learning_starts,
            replay_memory_size=self.replay_memory_size,
            batch_size=self.batch_size,
            model_path=self.output_path,
            embedding_network_weights_path=self.embedding_network_weights,
            n_groups=self.n_groups,
            fairness_constraints=self.fairness_constraints,
            use_reward_model=self.use_reward_model,
            no_cuda=self.no_cuda,
        )

        print("---------- Start Training")
        recommender.train(
            max_episode_num=self.max_eps_num, top_k=False, load_model=self.load_model
        )

        print("---------- Finish Training")
        print("---------- Saving Model")
        recommender.save_model(
            self.output()["actor_model"].path,
            self.output()["critic_model"].path,
            self.output()["srm_model"].path,
            os.path.join(self.output_path, "buffer.pkl"),
        )
        print("---------- Finish Saving Model")

    def evaluate_model(self, dataset):
        print("---------- Start Evaluation")

        item_groups_df = pd.DataFrame(
            dataset["item_groups"].items(), columns=["item_id", "group"]
        )
        catalog = item_groups_df.item_id.unique().tolist()

        top_k = [3, 5, 10, 15]

        _precision = []
        _propfair = []
        _ufg = []
        _recommended_item = []
        _random_recommended_item = []
        _exposure = []
        for k in top_k:
            sum_precision = 0
            sum_propfair = 0
            sum_reward = 0

            recommended_item = []
            random_recommended_item = []
            exposure = []

            env = ENV[self.algorithm](
                users_dict=dataset["eval_users_dict"],
                n_groups=self.n_groups,
                item_groups=dataset["item_groups"],
                items_metadata=dataset["items_metadata"],
                items_df=dataset["items_df"],
                state_size=self.state_size,
                done_count=k,
                fairness_constraints=self.fairness_constraints,
                reward_threshold=self.reward_threshold,
                reward_version=self.reward_version,
                use_only_reward_model=True,
                user_intent_threshold=self.user_intent_threshold,
                user_intent=self.user_intent,
                title_emb_path=dataset["title_emb"],
            )
            available_users = env.available_users

            recommender = AGENT[self.algorithm](
                env=env,
                is_test=True,
                users_num=self.users_num,
                items_num=self.items_num,
                srm_size=self.srm_size,
                srm_type="{}_{}".format(self.algorithm, self.reward_version),
                state_size=self.state_size,
                train_version=self.train_version,
                embedding_dim=self.embedding_dim,
                actor_hidden_dim=self.actor_hidden_dim,
                actor_learning_rate=self.actor_learning_rate,
                critic_hidden_dim=self.critic_hidden_dim,
                critic_learning_rate=self.critic_learning_rate,
                discount_factor=self.discount_factor,
                tau=self.tau,
                learning_starts=self.learning_starts,
                replay_memory_size=self.replay_memory_size,
                batch_size=self.batch_size,
                model_path=self.output_path,
                embedding_network_weights_path=self.embedding_network_weights,
                n_groups=self.n_groups,
                fairness_constraints=self.fairness_constraints,
                use_reward_model=self.use_reward_model,
                no_cuda=self.no_cuda,
            )

            for user_id in tqdm(available_users):

                eval_env = ENV[self.algorithm](
                    users_dict=dataset["eval_users_dict"],
                    n_groups=self.n_groups,
                    item_groups=dataset["item_groups"],
                    items_metadata=dataset["items_metadata"],
                    items_df=dataset["items_df"],
                    state_size=self.state_size,
                    done_count=k,
                    fairness_constraints=self.fairness_constraints,
                    reward_threshold=self.reward_threshold,
                    reward_version=self.reward_version,
                    user_intent_threshold=self.user_intent_threshold,
                    user_intent=self.user_intent,
                    use_only_reward_model=True,
                    reward_model=recommender.reward_model,
                    device=recommender.device,
                    fix_user_id=user_id,
                    title_emb_path=dataset["title_emb"],
                )

                result = recommender.online_evaluate(
                    top_k=False, load_model=True, env=eval_env
                )

                recommended_item.append(result["recommended_items"])
                random_recommended_item.append({user_id: sample(catalog, k)})
                exposure.append(result["exposure"])

                sum_precision += result["precision"]
                sum_propfair += result["propfair"]
                sum_reward += result["reward"]

                del eval_env

            _precision.append(sum_precision / len(dataset["eval_users_dict"]))
            _propfair.append(sum_propfair / len(dataset["eval_users_dict"]))
            _ufg.append(
                (sum_propfair / len(dataset["eval_users_dict"]))
                / (1 - (sum_precision / len(dataset["eval_users_dict"])))
            )
            _recommended_item.append(recommended_item)
            _random_recommended_item.append(random_recommended_item)
            _exposure.append(exposure)

        feature_df = pd.DataFrame(
            item_groups_df[["item_id"]]
            .apply(lambda x: recommender.get_items_emb(x).cpu().numpy().tolist())[
                "item_id"
            ]
            .tolist()
        )

        metrics = {}
        for k in range(len(top_k)):
            recs = pd.DataFrame(
                [i.values() for i in _recommended_item[k]], columns=["sorted_actions"]
            ).sorted_actions.values.tolist()

            exposure = np.array(_exposure[k]).mean(axis=0)
            ideal_exposure = self.fairness_constraints / np.sum(
                self.fairness_constraints
            )

            metrics[top_k[k]] = {
                "precision": round(_precision[k] * 100, 4),
                "propfair": round(_propfair[k] * 100, 4),
                "ufg": round(_ufg[k], 4),
                "coverage": round(rm.prediction_coverage(recs, catalog), 4),
                "personalization": round(rm.personalization(recs) * 100, 4),
                "intra_list_similarity": round(
                    rm.intra_list_similarity(recs, feature_df), 4
                ),
                "exposure": exposure.tolist(),
                "ideal_exposure": ideal_exposure.tolist(),
            }

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[i for i in range(1, self.n_groups + 1)],
                    y=exposure,
                    mode="lines+markers",
                    name="Group Exposure",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[i for i in range(1, self.n_groups + 1)],
                    y=ideal_exposure,
                    mode="lines+markers",
                    name="Ideal Exposure",
                )
            )
            fig.update_layout(
                title="Group Exposure vs Ideal Exposure",
                xaxis_title="Group",
                yaxis_title="Exposure",
            )
            fig.write_image(
                os.path.join(
                    self.output_path,
                    "group_exposure_vs_ideal_exposure_{}.png".format(k),
                )
            )

        with open(os.path.join(self.output_path, "metrics.json"), "w") as f:
            json.dump(metrics, f)

        print("---------- Finish Evaluation")

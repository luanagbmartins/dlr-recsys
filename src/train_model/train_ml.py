import os
import time
import datetime
import yaml
import json
import pickle
import luigi
import wandb
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


from src.data.dataset import DatasetGeneration
from src.environment.ml_env import OfflineEnv
from src.environment.ml_fair_env import OfflineEnv as OfflineFairEnv

from src.model.recommender import DRRAgent, FairRecAgent

AGENT = dict(drr=DRRAgent, fairrec=FairRecAgent)


class MovieLens(luigi.Task):
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
    top_k: int = luigi.IntParameter(default=10)
    embedding_network_weights: str = luigi.Parameter(default="")

    train_version: str = luigi.Parameter()
    use_wandb: bool = luigi.BoolParameter()
    load_model: bool = luigi.BoolParameter()
    dataset_path: str = luigi.Parameter()
    evaluate: bool = luigi.BoolParameter()

    def output(self):
        return {
            "actor_model": luigi.LocalTarget(
                os.path.join(self.output_path, "actor.h5")
            ),
            "critic_model": luigi.LocalTarget(
                os.path.join(self.output_path, "critic.h5")
            ),
        }

    def run(self):
        dataset = self.load_data()
        self.train_model(dataset)

        if self.evaluate:
            self.evaluate_model(dataset)

    def load_data(self):
        with open(self.dataset_path) as json_file:
            _dataset_path = json.load(json_file)

        dataset = {}
        with open(_dataset_path["train_users_dict"], "rb") as pkl_file:
            dataset["train_users_dict"] = pickle.load(pkl_file)
        with open(_dataset_path["train_users_history_lens"], "rb") as pkl_file:
            dataset["train_users_history_lens"] = pickle.load(pkl_file)
        with open(_dataset_path["eval_users_dict"], "rb") as pkl_file:
            dataset["eval_users_dict"] = pickle.load(pkl_file)
        with open(_dataset_path["eval_users_history_lens"], "rb") as pkl_file:
            dataset["eval_users_history_lens"] = pickle.load(pkl_file)
        with open(_dataset_path["users_history_lens"], "rb") as pkl_file:
            dataset["users_history_lens"] = pickle.load(pkl_file)
        with open(_dataset_path["movies_id_to_movies"], "rb") as pkl_file:
            dataset["movies_id_to_movies"] = pickle.load(pkl_file)
        with open(_dataset_path["movies_groups"], "rb") as pkl_file:
            dataset["movies_groups"] = pickle.load(pkl_file)

        return dataset

    def train_model(self, dataset):
        print("---------- Prepare Env")

        if self.algorithm == "drr":
            env = OfflineEnv(
                dataset["train_users_dict"],
                dataset["train_users_history_lens"],
                dataset["movies_id_to_movies"],
                dataset["movies_groups"],
                self.state_size,
            )
        else:
            env = OfflineFairEnv(
                dataset["train_users_dict"],
                dataset["train_users_history_lens"],
                dataset["movies_id_to_movies"],
                dataset["movies_groups"],
                self.state_size,
                self.fairness_constraints,
            )

        print("---------- Initialize Agent")
        recommender = AGENT[self.algorithm](
            env=env,
            users_num=self.users_num,
            items_num=self.items_num,
            srm_size=self.srm_size,
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
            embedding_network_weights_path=os.path.join(
                self.output_path, self.embedding_network_weights
            ),
            n_groups=self.n_groups,
            fairness_constraints=self.fairness_constraints,
        )

        print("---------- Build Networks")
        recommender.actor.build_networks()
        recommender.critic.build_networks()

        print("---------- Start Training")
        recommender.train(
            max_episode_num=self.max_eps_num, top_k=False, load_model=self.load_model
        )

        print("---------- Finish Training")
        print("---------- Saving Model")
        recommender.save_model(
            self.output()["actor_model"].path,
            self.output()["critic_model"].path,
        )
        print("---------- Finish Saving Model")

    def evaluate_model(self, dataset):
        actor_checkpoints = sorted(
            [
                int((f.split("_")[1]).split(".")[0])
                for f in os.listdir(self.output_path)
                if f.startswith("actor_")
            ]
        )
        critic_checkpoints = sorted(
            [
                int((f.split("_")[1]).split(".")[0])
                for f in os.listdir(self.output_path)
                if f.startswith("critic_")
            ]
        )

        for actor_checkpoint, critic_checkpoint in zip(
            actor_checkpoints, critic_checkpoints
        ):
            sum_precision = 0
            sum_ndcg = 0
            sum_propfair = 0
            sum_cvr = 0
            sum_ufg = 0

            if self.algorithm == "drr":
                env = OfflineEnv(
                    dataset["train_users_dict"],
                    dataset["train_users_history_lens"],
                    dataset["movies_id_to_movies"],
                    dataset["movies_groups"],
                    self.state_size,
                )
            else:
                env = OfflineFairEnv(
                    dataset["train_users_dict"],
                    dataset["train_users_history_lens"],
                    dataset["movies_id_to_movies"],
                    dataset["movies_groups"],
                    self.state_size,
                    self.fairness_constraints,
                )

            recommender = AGENT[self.algorithm](
                env=env,
                users_num=self.users_num,
                items_num=self.items_num,
                srm_size=self.srm_size,
                state_size=self.state_size,
                train_version=self.train_version,
                is_test=True,
                use_wandb=self.use_wandb,
                embedding_dim=self.embedding_dim,
                actor_hidden_dim=self.actor_hidden_dim,
                actor_learning_rate=self.actor_learning_rate,
                critic_hidden_dim=self.critic_hidden_dim,
                critic_learning_rate=self.critic_learning_rate,
                discount_factor=self.discount_factor,
                tau=self.tau,
                replay_memory_size=self.replay_memory_size,
                batch_size=self.batch_size,
                model_path=self.output_path,
                embedding_network_weights_path=os.path.join(
                    self.output_path, self.embedding_network_weights
                ),
                n_groups=self.n_groups,
                fairness_constraints=self.fairness_constraints,
            )

            recommender.actor.build_networks()
            recommender.critic.build_networks()
            recommender.load_model(
                os.path.join(self.output_path, "actor_{}.h5".format(actor_checkpoint)),
                os.path.join(
                    self.output_path, "critic_{}.h5".format(critic_checkpoint)
                ),
            )
            for user_id in dataset["eval_users_dict"].keys():
                eval_env = OfflineEnv(
                    dataset["eval_users_dict"],
                    dataset["users_history_lens"],
                    dataset["movies_id_to_movies"],
                    dataset["movies_groups"],
                    self.state_size,
                    fix_user_id=user_id,
                )

                recommender.env = eval_env

                precision, ndcg, propfair, cvr, ufg = recommender.evaluate(
                    eval_env, top_k=self.top_k
                )
                sum_precision += precision
                sum_ndcg += ndcg
                sum_propfair += propfair
                sum_cvr += cvr
                sum_ufg += ufg

                del eval_env

            print("---------- Evaluation")
            print("- precision@: ", sum_precision / len(dataset["eval_users_dict"]))
            print("- ndcg@: ", sum_ndcg / len(dataset["eval_users_dict"]))
            print("- propfair: ", sum_propfair / len(dataset["eval_users_dict"]))
            print("- cvr: ", sum_cvr / len(dataset["eval_users_dict"]))
            print("- ufg: ", sum_ufg / len(dataset["eval_users_dict"]))
            print()

            if self.use_wandb:
                wandb.log(
                    {
                        "cp_evaluation_precision@{}".format(self.top_k): sum_precision
                        / len(dataset["eval_users_dict"]),
                        "cp_evaluation_ndcg@{}".format(self.top_k): sum_ndcg
                        / len(dataset["eval_users_dict"]),
                        "cp_evaluation_propfair": sum_propfair
                        / len(dataset["eval_users_dict"]),
                        "cp_evaluation_cvr": sum_cvr / len(dataset["eval_users_dict"]),
                        "cp_evaluation_ufg": sum_ufg / len(dataset["eval_users_dict"]),
                    }
                )

            del recommender

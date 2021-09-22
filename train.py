import os
import time
import datetime
import yaml
import luigi
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


from src.data.dataset import DatasetGeneration
from src.environment.ml_env import OfflineEnv
from src.train_model import MovieLens, MLEmbedding

OUTPUT_PATH = os.path.join(os.getcwd(), "model")

TRAINER = dict(
    movie_lens_100k=MovieLens,
    movie_lens_100k_fair=MovieLens,
    movie_lens_1m=MovieLens,
    movie_lens_1m_fair=MovieLens,
)


class DRLTrain(luigi.Task):
    use_wandb: bool = luigi.BoolParameter()
    load_model: bool = luigi.BoolParameter()
    evaluate: bool = luigi.BoolParameter()
    train_version: str = luigi.Parameter(default="movie_lens_1m")
    dataset_version: str = luigi.Parameter(default="movie_lens_1m")
    train_id: str = luigi.Parameter(default="")

    def __init__(self, *args, **kwargs):
        super(DRLTrain, self).__init__(*args, **kwargs)

        self.dtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.output_path = os.path.join(
            OUTPUT_PATH,
            self.train_version,
            str(self.train_version + "_{}".format(self.dtime)),
        )
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "images"), exist_ok=True)

        if len(self.train_id) > 0:
            self.output_path = os.path.join(
                OUTPUT_PATH, self.train_version, self.train_id
            )

    def run(self):
        print("---------- Generate Dataset")
        dataset = yield DatasetGeneration(self.dataset_version)

        print("---------- Train Embedding")
        emb = yield MLEmbedding(
            **self.train_config["emb_train"],
            users_num=self.train_config["users_num"],
            items_num=self.train_config["items_num"],
            genres_num=self.train_config["genres_num"],
            embedding_dim=self.train_config["embedding_dim"],
            emb_model=self.train_config["emb_model"],
            output_path=self.output_path,
            train_version=self.train_version,
            dataset_path=dataset.path,
            use_wandb=self.use_wandb,
        )

        print("---------- Train Model")
        train = yield TRAINER[self.train_version](
            **self.train_config["model_train"],
            users_num=self.train_config["users_num"],
            items_num=self.train_config["items_num"],
            genres_num=self.train_config["genres_num"],
            embedding_dim=self.train_config["embedding_dim"],
            emb_model=self.train_config["emb_model"],
            embedding_network_weights=emb.path,
            output_path=self.output_path,
            train_version=self.train_version,
            use_wandb=self.use_wandb,
            load_model=self.load_model,
            dataset_path=dataset.path,
            evaluate=self.evaluate,
        )

    @property
    def train_config(self):
        path = os.path.abspath(
            os.path.join("model", "{}.yaml".format(self.train_version))
        )

        with open(path) as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)

        return train_config

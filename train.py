import os
import time
import yaml
import luigi
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


from src.data.dataset import DatasetGeneration
from src.environment.ml_env import OfflineEnv
from src.train_model import MovieLens

TRAINER = dict(movie_lens_100k=MovieLens, movie_lens_100k_fair=MovieLens)


class DRRTrain(luigi.Task):
    use_wandb: bool = luigi.BoolParameter()
    load_model: bool = luigi.BoolParameter()
    evaluate: bool = luigi.BoolParameter()
    train_version: str = luigi.Parameter(default="movie_lens_1m")
    dataset_version: str = luigi.Parameter(default="movie_lens_1m")

    def run(self):
        print("---------- Generate Dataset")
        dataset = yield DatasetGeneration(self.dataset_version)

        print("---------- Train Model")
        train = yield TRAINER[self.train_version](
            **self.train_config,
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

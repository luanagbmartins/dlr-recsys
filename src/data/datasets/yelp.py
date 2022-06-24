import os
import luigi
import pickle
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing


from ..utils import DownloadDataset


class YelpLoadAndPrepareDataset(luigi.Task):
    output_path: str = luigi.Parameter(default=os.path.join(os.getcwd(), "data/"))
    n_groups: int = luigi.IntParameter(default=4)

    def __init__(self, *args, **kwargs):
        super(YelpLoadAndPrepareDataset, self).__init__(*args, **kwargs)

        self.data_dir = os.path.join(self.output_path, "yelp")

    # def requires(self):
    #     return DownloadDataset(dataset="trivago", output_path=self.output_path)

    def output(self):
        return {
            "items_df": luigi.LocalTarget(os.path.join(self.data_dir, "items.csv")),
            "items_metadata": luigi.LocalTarget(
                os.path.join(self.data_dir, "items_metadata.csv")
            ),
            "ratings_df": luigi.LocalTarget(os.path.join(self.data_dir, "review.csv")),
            "train_users_dict": luigi.LocalTarget(
                os.path.join(self.data_dir, "train_users_dict.pkl")
            ),
            "train_users_history_lens": luigi.LocalTarget(
                os.path.join(self.data_dir, "train_users_history_lens.pkl")
            ),
            "eval_users_dict": luigi.LocalTarget(
                os.path.join(self.data_dir, "eval_users_dict.pkl")
            ),
            "eval_users_history_lens": luigi.LocalTarget(
                os.path.join(self.data_dir, "eval_users_history_lens.pkl")
            ),
            "users_history_lens": luigi.LocalTarget(
                os.path.join(self.data_dir, "users_history_lens.pkl")
            ),
            "item_groups": luigi.LocalTarget(
                os.path.join(self.data_dir, "item_groups.pkl")
            ),
        }

    def run(self):
        print("---------- Load Dataset")
        # datasets = self.load_dataset()

        print("---------- Prepare Dataset")
        # self.prepareDataset(datasets)

    def load_dataset(self):
        pass

    def prepareDataset(self, datasets):
        pass

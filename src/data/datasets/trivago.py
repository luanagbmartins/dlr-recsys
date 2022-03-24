import os
import luigi
import pickle
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing


from ..utils import DownloadDataset


class TrivagoLoadAndPrepareDataset(luigi.Task):
    output_path: str = luigi.Parameter(default=os.path.join(os.getcwd(), "data/"))
    n_groups: int = luigi.IntParameter(default=4)

    def __init__(self, *args, **kwargs):
        super(TrivagoLoadAndPrepareDataset, self).__init__(*args, **kwargs)

        self.data_dir = os.path.join(self.output_path, "trivago")

    def requires(self):
        return DownloadDataset(dataset="trivago", output_path=self.output_path)

    def output(self):
        return {
            "ratings_df": luigi.LocalTarget(os.path.join(self.data_dir, "ratings.csv")),
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
        datasets = self.load_dataset()

        print("---------- Prepare Dataset")
        self.prepareDataset(datasets)

    def load_dataset(self):

        df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        df_meta = pd.read_csv(os.path.join(self.data_dir, "item_metadata.csv"))

        recsys_cities = [
            "Lausanne, Switzerland",
            "New York, USA",
            "Barcelona, Spain",
            "Chicago, USA",
            "Dublin, Ireland",
            "Hong Kong, Hong Kong",
            "Vienna, Austria",
            "Boston, USA",
            "Como, Italy",
            "Vancouver, Canada",
            "Copenhagen, Denmark",
            "Rio de Janeiro, Brazil",
        ]

        df = df[(df["action_type"] == "clickout item")]
        df = df[df["city"].isin(recsys_cities)]
        df = df[["timestamp", "user_id", "reference", "impressions"]]
        df = df.sort_values("timestamp")

        df["impressions"] = df["impressions"].apply(lambda x: x.split("|"))
        df["reference"] = df["reference"].astype("int")
        df["impressions"] = df["impressions"].apply(lambda x: [int(i) for i in x])

        # df["pos"] = df[["reference", "impressions"]].apply(
        #     lambda x: x.impressions.index(x.reference)
        #     if x.reference in x.impressions
        #     else -1,
        #     axis=1,
        # )
        # df = df.drop(df[df["pos"] == -1].index)

        df = df.explode("impressions")
        df["clicked"] = df[["impressions", "reference"]].apply(
            lambda x: 1 if int(x["impressions"]) == int(x["reference"]) else 0, axis=1
        )

        df = df[["user_id", "impressions", "clicked"]]
        df = df.rename(columns={"impressions": "item_id"})

        item_encoder = preprocessing.LabelEncoder().fit(df.item_id.values)
        df.item_id = item_encoder.transform(df.item_id.values)

        user_encoder = preprocessing.LabelEncoder().fit(df.user_id.values)
        df.user_id = user_encoder.transform(df.user_id.values)

        df = (
            df[["user_id", "item_id", "clicked"]]
            .groupby(["user_id", "item_id"])
            .max()
            .reset_index()
        )

        # Save preprocessed dataframes
        datasets = {"ratings": df}
        for dataset in datasets:
            datasets[dataset].to_csv(
                os.path.join(self.data_dir, str(dataset + ".csv")),
                index=False,
            )

    def prepareDataset(self, datasets):
        users_dict = {user: [] for user in set(datasets["ratings"]["user_id"])}

        ratings_df_gen = datasets["ratings"].iterrows()
        users_dict_for_history_len = {
            user: [] for user in set(datasets["ratings"]["user_id"])
        }
        for data in ratings_df_gen:
            users_dict[data[1]["user_id"]].append(
                (data[1]["item_id"], data[1]["rating"])
            )
            if data[1]["rating"] >= 1:
                users_dict_for_history_len[data[1]["user_id"]].append(
                    (data[1]["item_id"], data[1]["rating"])
                )
        users_history_lens = [
            len(users_dict_for_history_len[u])
            for u in set(datasets["ratings"]["user_id"])
        ]

        users_num = max(datasets["ratings"]["user_id"]) + 1
        items_num = max(datasets["ratings"]["item_id"]) + 1

        print(users_num, items_num)

        # items groups
        z = np.random.geometric(p=0.35, size=items_num)
        w = z % self.n_groups
        w = [i if i > 0 else self.n_groups for i in w]
        item_groups = {i: w[i] for i in range(items_num)}

        # Training setting
        train_users_num = int(users_num * 0.8)
        train_users_dict = {k: users_dict.get(k) for k in range(0, train_users_num + 1)}
        train_users_history_lens = users_history_lens[:train_users_num]

        # Evaluating setting
        eval_users_num = int(users_num * 0.2)
        eval_users_dict = {
            k: users_dict[k] for k in range(users_num - eval_users_num, users_num)
        }
        eval_users_history_lens = users_history_lens[-eval_users_num:]

        with open(self.output()["train_users_dict"].path, "wb") as file:
            pickle.dump(train_users_dict, file)

        with open(self.output()["train_users_history_lens"].path, "wb") as file:
            pickle.dump(train_users_history_lens, file)

        with open(self.output()["eval_users_dict"].path, "wb") as file:
            pickle.dump(eval_users_dict, file)

        with open(self.output()["eval_users_history_lens"].path, "wb") as file:
            pickle.dump(eval_users_history_lens, file)

        with open(self.output()["users_history_lens"].path, "wb") as file:
            pickle.dump(users_history_lens, file)

        with open(self.output()["item_groups"].path, "wb") as file:
            pickle.dump(item_groups, file)

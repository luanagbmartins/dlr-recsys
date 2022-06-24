import os
import luigi
import pickle
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing

from ..utils import DownloadDataset


class ML1MLoadAndPrepareDataset(luigi.Task):
    output_path: str = luigi.Parameter(default=os.path.join(os.getcwd(), "data/"))
    n_groups: int = luigi.IntParameter(default=4)

    def __init__(self, *args, **kwargs):
        super(ML1MLoadAndPrepareDataset, self).__init__(*args, **kwargs)

        self.data_dir = os.path.join(self.output_path, "ml-1m")

    def requires(self):
        return DownloadDataset(dataset="ml-1m", output_path=self.output_path)

    def output(self):
        return {
            "items_df": luigi.LocalTarget(os.path.join(self.data_dir, "movies.csv")),
            "items_metadata": luigi.LocalTarget(
                os.path.join(self.data_dir, "items_metadata.csv")
            ),
            "users_df": luigi.LocalTarget(os.path.join(self.data_dir, "users.csv")),
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

        ratings_list = [
            i.strip().split("::")
            for i in open(os.path.join(self.data_dir, "ratings.dat"), "r").readlines()
        ]
        users_list = [
            i.strip().split("::")
            for i in open(os.path.join(self.data_dir, "users.dat"), "r").readlines()
        ]
        movies_list = [
            i.strip().split("::")
            for i in open(
                os.path.join(self.data_dir, "movies.dat"), encoding="latin-1"
            ).readlines()
        ]

        ratings_df = pd.DataFrame(
            ratings_list,
            columns=["user_id", "item_id", "rating", "timestamp"],
            dtype=np.uint32,
        )

        movies_df = pd.DataFrame(
            movies_list, columns=["item_id", "item_name", "genres"]
        )
        movies_df["item_id"] = movies_df["item_id"].apply(pd.to_numeric)

        users_df = pd.DataFrame(
            users_list, columns=["user_id", "gender", "age", "occupation", "zip_code"]
        )

        ratings_df["user_id"] = ratings_df["user_id"].astype("int")
        ratings_df["item_id"] = ratings_df["item_id"].astype("int")
        users_df["user_id"] = users_df["user_id"].astype("int")
        movies_df["item_id"] = movies_df["item_id"].astype("int")

        # Encode target labels with value between 0 and n_classes-1
        movies_encoder = preprocessing.LabelEncoder()
        movies_encoder.fit(movies_df["item_id"].values)
        movies_df["item_id"] = movies_encoder.transform(movies_df["item_id"].values)
        ratings_df["item_id"] = movies_encoder.transform(ratings_df["item_id"].values)

        users_encoder = preprocessing.LabelEncoder()
        users_encoder.fit(users_df["user_id"].values)
        users_df["user_id"] = users_encoder.transform(users_df["user_id"].values)
        ratings_df["user_id"] = users_encoder.transform(ratings_df["user_id"].values)

        # Getting series of lists by applying split operation.
        movies_df.genres = movies_df.genres.str.split("|")
        # Getting distinct genre types for generating columns of genre type.
        genre_columns = list(set([j for i in movies_df["genres"].tolist() for j in i]))

        for j in genre_columns:
            movies_df[j] = 0
        for i in range(movies_df.shape[0]):
            for j in genre_columns:
                if j in movies_df["genres"].iloc[i]:
                    movies_df.loc[i, j] = 1

        # dropping 'genre' columns as it has already been one hot encoded.
        movies_df.drop("genres", axis=1, inplace=True)

        items_metadata = movies_df.drop(columns=["item_name"])

        movies_df = movies_df[["item_id", "item_name"]]

        # Save preprocessed dataframes
        datasets = {
            "ratings": ratings_df,
            "movies": movies_df,
            "users": users_df,
            "items_metadata": items_metadata,
        }

        for dataset in datasets:
            datasets[dataset].to_csv(
                os.path.join(self.data_dir, str(dataset + ".csv")),
                index=False,
            )

        return datasets

    def prepareDataset(self, datasets):

        datasets["ratings"] = datasets["ratings"].sort_values("timestamp")
        datasets["ratings"] = datasets["ratings"].applymap(int)

        users_dict = {user: [] for user in set(datasets["ratings"]["user_id"])}

        ratings_df_gen = datasets["ratings"].iterrows()
        users_dict_for_history_len = {
            user: [] for user in set(datasets["ratings"]["user_id"])
        }
        for data in ratings_df_gen:
            users_dict[data[1]["user_id"]].append(
                (data[1]["item_id"], data[1]["rating"])
            )
            if data[1]["rating"] >= 4:
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

        # # items groups
        # z = np.random.geometric(p=0.35, size=items_num)
        # w = z % self.n_groups
        # w = [i if i > 0 else self.n_groups for i in w]
        # item_groups = {i: w[i] for i in range(items_num)}

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

        # with open(self.output()["item_groups"].path, "wb") as file:
        #     pickle.dump(item_groups, file)

    def _split_and_index(self, string):
        genres = [
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]
        string = string.split("|")
        for i, s in enumerate(string):
            string[i] = genres.index(s)
        return string

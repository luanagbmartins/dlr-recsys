import os
import abc
import math
import luigi
import zipfile
import bz2
import requests
import pandas as pd
from tqdm import tqdm

OUTPUT_PATH = os.path.join(os.getcwd(), "data/")
DATASETS = {
    "ml-100k": {
        "type": "url",
        "download": ["https://files.grouplens.org/datasets/movielens/ml-100k.zip"],
    },
    "ml-1m": {
        "type": "url",
        "download": ["https://files.grouplens.org/datasets/movielens/ml-1m.zip"],
    },
    "ml-10m": {
        "type": "url",
        "download": ["https://files.grouplens.org/datasets/movielens/ml-10m.zip"],
    },
    "ml-20m": {
        "type": "url",
        "download": ["https://files.grouplens.org/datasets/movielens/ml-20m.zip"],
    },
    "ml-25m": {
        "type": "url",
        "download": ["https://files.grouplens.org/datasets/movielens/ml-25m.zip"],
    },
}


def split_train_test(data, train_ratio=0.8):
    # Lista para armazenar os subsets de treino e teste
    train_list = []
    test_list = []

    for _, group in data.groupby("user_id"):
        # Ordena as interações por timestamp
        group = group.sort_values("timestamp")

        # Calcula o ponto de corte para o treino (80% das interações)
        split_point = math.ceil(len(group) * train_ratio)

        # Separa o conjunto de treino e teste
        train_list.append(group.iloc[:split_point])
        test_list.append(group.iloc[split_point:])

    # Concatena todos os subsets de treino e teste
    train_data = pd.concat(train_list)
    test_data = pd.concat(test_list)

    return train_data, test_data


class DownloadDataset(luigi.Task, metaclass=abc.ABCMeta):
    output_path: str = luigi.Parameter(default=OUTPUT_PATH)
    dataset: str = luigi.ChoiceParameter(choices=DATASETS.keys())

    def output(self):
        return [
            luigi.LocalTarget(
                os.path.join(self.output_path, self.dataset, os.path.basename(p))
            )
            for p in DATASETS[self.dataset]["download"]
        ]

    def run(self):
        if DATASETS[self.dataset]["type"] == "url":
            self.download_url(self.dataset, output_path=self.output_path)
        elif DATASETS[self.dataset]["type"] == "kaggle":
            self.download_kaggle(self.dataset, output_path=self.output_path)

    def download_url(self, name, cache=True, output_path=".", **kws):
        results = []
        for url in DATASETS[name]["download"]:
            output_file = os.path.join(output_path, name, os.path.basename(url))
            if not os.path.isfile(output_file) or not cache:
                # Streaming, so we can iterate over the response.
                r = requests.get(url, stream=True)

                # Total size in bytes.
                total_size = int(r.headers.get("content-length", 0))
                block_size = 1024
                wrote = 0
                os.makedirs(os.path.split(output_file)[0], exist_ok=True)
                with open(output_file, "wb") as f:
                    for data in tqdm(
                        r.iter_content(block_size),
                        total=math.ceil(total_size // block_size),
                        unit="KB",
                        unit_scale=True,
                    ):
                        wrote = wrote + len(data)
                        f.write(data)
                if total_size != 0 and wrote != total_size:
                    raise ConnectionError("ERROR, something went wrong")

            if output_file.endswith(".zip"):
                with zipfile.ZipFile(output_file, "r") as zip_ref:
                    zip_ref.extractall(output_path)

            if output_file.endswith(".bz2"):
                _zipfile = bz2.BZ2File(output_file)
                data = _zipfile.read()
                newfilepath = output_file[:-4]
                open(newfilepath, "wb").write(data)

    def download_kaggle(self, name, output_path=".", **kws):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise ImportError(
                "Could not find kaggle.json. Please download the Kaggle API token to continue."
            )

        api = KaggleApi()
        api.authenticate()
        for dataset in DATASETS[name]["download"]:
            api.dataset_download_files(
                dataset,
                path=os.path.join(output_path, name),
                unzip=True,
            )

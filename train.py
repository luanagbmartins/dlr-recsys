import os
import datetime
import yaml
import luigi

from src.train_model import RSRL
from src.data.dataset import DatasetGeneration


OUTPUT_PATH = os.path.join(os.getcwd(), "model")


class TrainRS(luigi.Task):
    """Recommendation system training module"""

    use_wandb: bool = luigi.BoolParameter()
    load_model: bool = luigi.BoolParameter()
    evaluate: bool = luigi.BoolParameter()
    train_version: str = luigi.Parameter()
    dataset_version: str = luigi.Parameter()
    train_id: str = luigi.Parameter(default="")

    def __init__(self, *args, **kwargs):
        super(TrainRS, self).__init__(*args, **kwargs)

        if len(self.train_id) > 0:
            self.output_path = os.path.join(
                OUTPUT_PATH, self.train_version, self.train_id
            )
        else:
            dtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_path = os.path.join(
                OUTPUT_PATH,
                self.train_version,
                str(self.train_version + "_{}".format(dtime)),
            )
            os.makedirs(self.output_path, exist_ok=True)
            os.makedirs(os.path.join(self.output_path, "images"), exist_ok=True)

            path = os.path.abspath(
                os.path.join("model", "{}.yaml".format(self.train_version))
            )
            with open(path) as f:
                train_config = yaml.load(f, Loader=yaml.FullLoader)

            with open(
                os.path.join(self.output_path, "{}.yaml".format(self.train_version)),
                "w",
            ) as file:
                yaml.dump(train_config, file)

    def run(self):
        print("---------- Generate Dataset")
        dataset = yield DatasetGeneration(self.dataset_version)

        print("---------- Train Model")
        yield RSRL(
            **self.train_config["model_train"],
            algorithm=self.train_config["algorithm"],
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
            os.path.join(self.output_path, "{}.yaml".format(self.train_version))
        )

        with open(path) as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)

        return train_config

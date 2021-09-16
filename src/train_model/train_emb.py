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


from src.model.embedding import UserMovieEmbedding


class MLEmbedding(luigi.Task):
    output_path: str = luigi.Parameter()
    train_version: str = luigi.Parameter()
    dataset_path: str = luigi.Parameter()
    use_wandb: bool = luigi.BoolParameter()

    max_epoch: int = luigi.IntParameter(default=150)
    init_user_batch_size: int = luigi.IntParameter(default=32)
    final_user_batch_size: int = luigi.IntParameter(default=1024)
    users_num: int = luigi.IntParameter(default=943)
    items_num: int = luigi.IntParameter(default=1682)
    embedding_dim: int = luigi.IntParameter(default=50)

    def __init__(self, *args, **kwargs):
        super(MLEmbedding, self).__init__(*args, **kwargs)

        if self.use_wandb:
            wandb.init(
                project=self.train_version,
                config={
                    "users_num": self.users_num,
                    "items_num": self.items_num,
                    "embedding_dim": self.embedding_dim,
                },
            )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(self.output_path, "user_movie_at_once.h5")
        )

    def run(self):
        dataset = self.load_data()
        self.train_model(dataset)

    def load_data(self):
        with open(self.dataset_path) as json_file:
            _dataset_path = json.load(json_file)

        dataset = {}
        dataset["movies_df"] = pd.read_csv(_dataset_path["movies_df"])
        dataset["ratings_df"] = pd.read_csv(_dataset_path["ratings_df"])

        dataset["movies_df"]["movie_id"] = dataset["movies_df"]["movie_id"] - 1
        dataset["ratings_df"]["movie_id"] = dataset["ratings_df"]["movie_id"] - 1
        dataset["ratings_df"]["user_id"] = dataset["ratings_df"]["user_id"] - 1

        return dataset

    def train_model(self, dataset):
        user_movie_rating_df = dataset["ratings_df"][
            ["user_id", "movie_id", "rating"]
        ].apply(np.int32)
        modified_user_movie_rating_df = user_movie_rating_df.apply(np.int32)
        index_names = modified_user_movie_rating_df[
            modified_user_movie_rating_df["rating"] < 4
        ].index
        modified_user_movie_rating_df = modified_user_movie_rating_df.drop(index_names)
        modified_user_movie_rating_df = modified_user_movie_rating_df.drop(
            "rating", axis=1
        )
        u_m_pairs = modified_user_movie_rating_df.to_numpy()
        positive_user_movie_dict = {
            u: [] for u in range(0, max(modified_user_movie_rating_df["user_id"]) + 1)
        }
        for data in modified_user_movie_rating_df.iterrows():
            positive_user_movie_dict[data[1][0]].append(data[1][1])

        model = UserMovieEmbedding(self.users_num, self.items_num, self.embedding_dim)
        model([np.zeros((1)), np.zeros((1))])
        print(model.summary())

        optimizer = tf.keras.optimizers.Adam()
        bce = tf.keras.losses.BinaryCrossentropy()

        train_loss = tf.keras.metrics.Mean(name="train_loss")
        train_accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")

        for epoch in range(self.max_epoch):

            batch_size = self.init_user_batch_size * (epoch + 1)
            if batch_size > self.final_user_batch_size:
                batch_size = self.final_user_batch_size
            test_generator = self.generate_user_movie_batch(
                u_m_pairs,
                batch_size,
                modified_user_movie_rating_df,
                positive_user_movie_dict,
            )

            for step in range(len(user_movie_rating_df) // batch_size):
                u_batch, m_batch, u_m_label_batch = next(test_generator)

                self.train_step(
                    model,
                    bce,
                    optimizer,
                    [u_batch, m_batch],
                    u_m_label_batch,
                    train_loss,
                    train_accuracy,
                )

                print(
                    f"{epoch} epoch, Batch size : {batch_size}, \
                        {step} steps, Loss: {train_loss.result():0.4f}, \
                            Accuracy: {train_accuracy.result() * 100:0.1f}",
                    end="\r",
                )

                if self.use_wandb:
                    wandb.log(
                        {
                            "loss": train_loss.result(),
                            "accuracy": train_accuracy.result() * 100,
                        }
                    )

        model.save_weights(self.output().path)

    def generate_user_movie_batch(
        self,
        positive_pairs,
        batch_size,
        user_movie_rating,
        positive_user_movie_dict,
        negative_ratio=0.5,
    ):
        batch = np.zeros((batch_size, 3))
        positive_batch_size = batch_size - int(batch_size * negative_ratio)
        max_user_id = max(user_movie_rating["user_id"]) + 1
        max_movie_id = max(user_movie_rating["movie_id"]) + 1

        while True:
            idx = np.random.choice(len(positive_pairs), positive_batch_size)
            data = positive_pairs[idx]
            for i, d in enumerate(data):
                batch[i] = (d[0], d[1], 1)

            while i + 1 < batch_size:
                u = np.random.randint(1, max_user_id)
                m = np.random.randint(1, max_movie_id)
                if m not in positive_user_movie_dict[u]:
                    i += 1
                    batch[i] = (u, m, 0)

            np.random.shuffle(batch)
            yield batch[:, 0], batch[:, 1], batch[:, 2]

    @tf.function
    def train_step(
        self, model, bce, optimizer, inputs, labels, train_loss, train_accuracy
    ):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = bce(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

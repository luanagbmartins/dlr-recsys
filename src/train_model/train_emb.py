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


from src.model.embedding import (
    UserMovieEmbedding,
    MovieGenreEmbedding,
    UserMovieGenreEmbedding,
)


class MLEmbedding(luigi.Task):
    output_path: str = luigi.Parameter()
    train_version: str = luigi.Parameter()
    dataset_path: str = luigi.Parameter()
    use_wandb: bool = luigi.BoolParameter()

    emb_model: str = luigi.Parameter(default="user_movie")
    max_epoch: int = luigi.IntParameter(default=150)
    init_user_batch_size: int = luigi.IntParameter(default=32)
    final_user_batch_size: int = luigi.IntParameter(default=1024)
    movie_batch_size: int = luigi.IntParameter(default=128)
    users_num: int = luigi.IntParameter(default=943)
    items_num: int = luigi.IntParameter(default=1682)
    genres_num: int = luigi.IntParameter(default=0)
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
        return luigi.LocalTarget(os.path.join(self.output_path, "emb_models.json"))

    def run(self):
        dataset = self.load_data()

        if self.emb_model == "user_movie":
            self.train_user_movie(dataset)
        else:
            self.train_user_movie_genre(dataset)

    def load_data(self):
        with open(self.dataset_path) as json_file:
            _dataset_path = json.load(json_file)

        dataset = {}
        dataset["movies_df"] = pd.read_csv(_dataset_path["movies_df"])
        dataset["ratings_df"] = pd.read_csv(_dataset_path["ratings_df"])
        with open(_dataset_path["movies_genres_id"], "rb") as pkl_file:
            dataset["movies_genres_id"] = pickle.load(pkl_file)
        return dataset

    def train_user_movie_genre(self, dataset):
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

        positive_m_g_pairs = []
        negative_m_g_pairs = []
        for movie in dataset["movies_df"]["movie_id"]:
            for i in range(len(genres)):
                if i in dataset["movies_genres_id"][movie]:
                    positive_m_g_pairs.append((movie, i, 1))
                else:
                    negative_m_g_pairs.append((movie, i, 0))

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

        movie_step_per_epoch = len(positive_m_g_pairs) // self.movie_batch_size

        m_g_model = MovieGenreEmbedding(
            self.items_num, self.genres_num, self.embedding_dim
        )
        m_g_model([np.zeros((1)), np.zeros((1))])
        print(m_g_model.summary())

        optimizer = tf.keras.optimizers.Adam()
        bce = tf.keras.losses.BinaryCrossentropy()

        m_g_train_loss = tf.keras.metrics.Mean(name="train_loss")
        m_g_train_accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")

        for epoch in range(self.max_epoch):

            m_g_generator = self.generate_movie_genre_batch(
                positive_m_g_pairs,
                negative_m_g_pairs,
                self.movie_batch_size,
                negative_ratio=0.5,
            )

            for step in range(movie_step_per_epoch):
                m_batch, g_batch, m_g_label_batch = next(m_g_generator)
                self.mg_train_step(
                    m_g_model,
                    bce,
                    optimizer,
                    [m_batch, g_batch],
                    m_g_label_batch,
                    m_g_train_loss,
                    m_g_train_accuracy,
                )

                print(
                    f"{epoch} epoch, \
                        {step} steps, \
                            Loss: {m_g_train_loss.result():0.4f}, \
                                Accuracy: {m_g_train_accuracy.result() * 100:0.1f}",
                    end="\r",
                )

                if self.use_wandb:
                    wandb.log(
                        {
                            "loss": m_g_train_loss.result(),
                            "accuracy": m_g_train_accuracy.result() * 100,
                        }
                    )

        u_m_model = UserMovieGenreEmbedding(self.users_num, self.embedding_dim)
        u_m_model([np.zeros((1)), np.zeros((1, self.embedding_dim))])
        print(u_m_model.summary())

        optimizer = tf.keras.optimizers.Adam()
        bce = tf.keras.losses.BinaryCrossentropy()

        u_m_train_loss = tf.keras.metrics.Mean(name="train_loss")
        u_m_train_accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")

        for epoch in range(self.max_epoch):

            batch_size = self.init_user_batch_size * (epoch + 1)
            if batch_size > self.final_user_batch_size:
                batch_size = self.final_user_batch_size
            u_m_generator = self.generate_user_movie_batch(
                u_m_pairs,
                batch_size,
                modified_user_movie_rating_df,
                positive_user_movie_dict,
            )
            for step in range(len(user_movie_rating_df) // batch_size):
                u_batch, m_batch, u_m_label_batch = next(u_m_generator)
                # m_batch = m_g_model.get_layer("movie_embedding")(m_batch)

                genres = []
                for item in m_batch:
                    genres.append(dataset["movies_genres_id"][item])

                items_eb = m_g_model.get_layer("movie_embedding")(np.array(m_batch))
                genres_eb = []
                for items in m_batch:
                    ge = m_g_model.get_layer("genre_embedding")(
                        np.array(dataset["movies_genres_id"][items])
                    )
                    genres_eb.append(ge)
                genre_mean = []
                for g in genres_eb:
                    genre_mean.append(tf.reduce_mean(g / self.embedding_dim, axis=0))
                genre_mean = tf.stack(genre_mean)

                m_batch = tf.add(items_eb, genre_mean)
                self.um_train_step(
                    u_m_model,
                    bce,
                    optimizer,
                    [u_batch, m_batch],
                    u_m_label_batch,
                    u_m_train_loss,
                    u_m_train_accuracy,
                )

                print(
                    f"{epoch} epoch, \
                        {step} steps, \
                            Loss: {u_m_train_loss.result():0.4f}, \
                                Accuracy: {u_m_train_accuracy.result() * 100:0.1f}",
                    end="\r",
                )

                if self.use_wandb:
                    wandb.log(
                        {
                            "loss": u_m_train_loss.result(),
                            "accuracy": u_m_train_accuracy.result() * 100,
                        }
                    )

        m_g_model.save_weights(os.path.join(self.output_path, "movie_genre.h5"))
        u_m_model.save_weights(os.path.join(self.output_path, "user_movie_genre.h5"))
        _output = {
            "movie_genre": os.path.join(self.output_path, "movie_genre.h5"),
            "user_movie_genre": os.path.join(self.output_path, "user_movie_genre.h5"),
        }
        with open(self.output().path, "w") as file:
            json.dump(_output, file)

    def train_user_movie(self, dataset):
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

                self.um_train_step(
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

        model.save_weights(os.path.join(self.output_path, "user_movie.h5"))
        _output = {
            "user_movie": os.path.join(self.output_path, "user_movie.h5"),
        }
        with open(self.output().path, "w") as file:
            json.dump(_output, file)

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

    def generate_movie_genre_batch(
        self, positive_pairs, negative_pairs, batch_size, negative_ratio=0.5
    ):
        batch = np.zeros((batch_size, 3))
        num_of_positive = batch_size - int(batch_size * negative_ratio)

        while True:
            idx = np.random.choice(len(positive_pairs), num_of_positive)
            positive_data = np.array(positive_pairs)[idx]
            for i, data in enumerate(positive_data):
                batch[i] = data

            idx = np.random.choice(
                len(negative_pairs), int(batch_size * negative_ratio)
            )
            negative_data = np.array(negative_pairs)[idx]
            for i, data in enumerate(negative_data):
                batch[num_of_positive + i] = data

            np.random.shuffle(batch)
            yield batch[:, 0], batch[:, 1], batch[:, 2]

    @tf.function
    def um_train_step(
        self, model, bce, optimizer, inputs, labels, train_loss, train_accuracy
    ):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = bce(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def mg_train_step(
        self, model, bce, optimizer, inputs, labels, train_loss, train_accuracy
    ):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = bce(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

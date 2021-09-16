import tensorflow as tf
import numpy as np


class DRRAveStateRepresentation(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = tf.keras.layers.Conv1D(1, 1, 1)
        self.concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        items_eb = tf.transpose(x[1], perm=(0, 2, 1)) / self.embedding_dim
        wav = self.wav(items_eb)
        wav = tf.transpose(wav, perm=(0, 2, 1))
        wav = tf.squeeze(wav, axis=1)
        user_wav = tf.keras.layers.multiply([x[0], wav])
        concat = self.concat([x[0], user_wav, wav])
        return self.flatten(concat)


class FairRecStateRepresentation(tf.keras.Model):
    def __init__(self, embedding_dim, n_groups):
        super(FairRecStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = tf.keras.layers.Conv1D(1, 1, 1)
        self.add = tf.keras.layers.Add()

        self.attention = tf.keras.layers.Attention()

        self.fav = tf.keras.layers.Dense(embedding_dim, activation="relu")

        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        group_mean = []
        for group in x[1]:
            group_mean.append(tf.reduce_mean(group / self.embedding_dim, axis=0))
        group_mean = tf.stack(group_mean)
        items = self.add([x[0], np.expand_dims(group_mean, axis=0)])

        # attention_hidden_layer = tf.nn.relu(items)
        # score = self.V(attention_hidden_layer)
        # attention_weights = tf.nn.softmax(score, axis=1)
        # ups = attention_weights * items
        ups = self.attention([items, items])

        ups = tf.transpose(ups, perm=(0, 2, 1))  # / self.embedding_dim
        ups = self.wav(ups)
        ups = tf.transpose(ups, perm=(0, 2, 1))
        fs = self.fav(x[2])

        concat = self.concat([ups, np.expand_dims(fs, axis=0)])
        return self.flatten(concat)

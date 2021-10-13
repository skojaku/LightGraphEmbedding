"""This module is a wrapper for glove_python_binary."""

import numpy as np

from lime import random_walk_sampler
from lime.Base import NodeEmbedding

try:
    import glove
except ImportError:
    print(
        "Ignore this message if you do not use Glove. Otherwise, install glove python package by 'pip install glove_python_binary' "
    )


class Glove(NodeEmbedding):
    def __init__(
        self,
        num_walks=10,
        walk_length=40,
        window_length=10,
        restart_prob=0,
        p=1.0,
        q=1.0,
        verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.sampler = random_walk_sampler.SimpleWalkSampler(
            num_walks,
            walk_length,
            window_length,
            restart_prob,
            p,
            q,
            sample_center_context_pairs=True,
            verbose=False,
        )
        self.learning_rate = 0.05
        self.w2vparams = {"epochs": 25, "no_threads": 4}

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.sampler.sampling(A)
        center, context, freq = self.sampler.get_center_context_pairs()
        center = center.astype(int)
        context = context.astype(int)
        N = self.sampler.num_nodes
        self.cooccur = sparse.coo_matrix(
            (freq, (center, context)), shape=(N, N), dtype="double"
        )
        return self

    def transform(self, dim, return_out_vector=False):
        # Update the in-vector and out-vector if
        # (i) this is the first to compute the vectors or
        # (ii) the dimension is different from that
        # for the previous call of transform function
        update_embedding = False
        if self.out_vec is None:
            update_embedding = True
        elif self.out_vec.shape[1] != dim:
            update_embedding = True

        # Update the dimension and train the model
        if update_embedding:
            self.model = glove.Glove(
                no_components=dim, learning_rate=self.learning_rate
            )
            self.model.fit(self.cooccur, **self.w2vparams)
            self.in_vec = self.model.word_vectors
            self.out_vec = self.model.word_vectors

        if return_out_vector:
            return self.out_vec
        else:
            return self.in_vec

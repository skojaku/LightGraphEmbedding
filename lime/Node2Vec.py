"""Module for node2vec.

Grover, Aditya, and Jure Leskovec. 2016. “Node2Vec: Scalable Feature
Learning for Networks.” In Proceedings of the 22Nd ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining, 855–64.
KDD ’16. New York, NY, USA: ACM.
"""

import gensim
import numpy as np
import pandas as pd
from scipy import sparse

from lime.Base import NodeEmbedding
from lime.random_walk_sampler import RWSentenceGenerator


class Node2Vec(NodeEmbedding):
    """A python class for the node2vec.

    Parameters
    ----------
    num_walks : int (optional, default 10)
        Number of walks per node
    walk_length : int (optional, default 40)
        Length of walks
    window_length : int (optional, default 10)
    restart_prob : float (optional, default 0)
        Restart probability of a random walker.
    """

    def __init__(
        self,
        num_walks=10,
        walk_length=40,
        window_length=10,
        restart_prob=0,
        verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.restart_prob = restart_prob
        self.window_length = window_length

        self.sentences = None
        self.model = None
        self.verbose = verbose

        self.w2vparams = {
            "sg": 1,
            "min_count": 0,
            "epochs": self.num_walks,
            "workers": 24,
            "window": self.window_length,
        }

    def fit(self, net):
        """Estimating the parameters for embedding.

        Parameters
        ---------
        net : nx.Graph object
            Network to be embedded. The graph type can be anything if
            the graph type is supported for the node samplers.

        Return
        ------
        self : Node2Vec
        """
        self.sentence_generator = RWSentenceGenerator(
            net,
            num_walks=1,  # self.num_walks (because I set epochs=num_walks),
            walk_length=self.walk_length,
            restart_prob=self.restart_prob,
        )
        return self

    def update_embedding(self, dim):
        self.w2vparams["vector_size"] = dim
        self.model = gensim.models.Word2Vec(
            sentences=self.sentence_generator, **self.w2vparams
        )

        num_nodes = len(self.model.wv.index_to_key)
        self.in_vec = np.zeros((num_nodes, dim))
        self.out_vec = np.zeros((num_nodes, dim))
        for i in range(num_nodes):
            if "%d" % i not in self.model.wv:
                continue
            self.in_vec[i, :] = self.model.wv["%d" % i]
            self.out_vec[i, :] = self.model.syn1neg[
                self.model.wv.key_to_index["%d" % i]
            ]

import shutil
import unittest

import networkx as nx
import numpy as np
from scipy import sparse

import lime


class TestCalc(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(self.G)

    def test_node2vec(self):
        model = lime.Node2Vec()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32

    def test_deepwalk(self):
        model = lime.DeepWalk()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32

    def test_modularity(self):
        model = lime.ModularitySpectralEmbedding()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32

    def test_adjacency(self):
        model = lime.AdjacencySpectralEmbedding()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32

    def test_non_back_tracking(self):
        model = lime.NonBackTrackingSpectralEmbedding()
        model.fit(self.A)
        emb = model.transform(dim=32)

        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == 32


if __name__ == "__main__":
    unittest.main()

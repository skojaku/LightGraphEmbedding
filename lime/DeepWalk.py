"""Module for DeepWalk.

Perozzi, Bryan, Rami Al-Rfou, and Steven Skiena. 2014. “DeepWalk: Online Learning of Social Representations.” In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 701–10. KDD ’14. New York, NY, USA: Association for Computing Machinery.

"""
from Node2Vec import Node2Vec


class DeepWalk(Node2Vec):
    def __init__(self, **params):
        Node2Vec.__init__(self, **params)
        self.w2vparams["sg"] = 0
        self.w2vparams["hs"] = 1

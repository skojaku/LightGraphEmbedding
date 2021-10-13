"""This module contains the base classses for graph embedding classes"""

#
# Base class
#
class NodeEmbedding:
    """Super class for node embedding class."""

    def __init__(self):
        self.in_vec = None
        self.out_vec = None

    def fit(self):
        """Estimating the parameters for embedding."""
        pass

    def transform(self, dim, return_out_vector=False):
        """Compute the coordinates of nodes in the embedding space of the
        prescribed dimensions."""
        # Update the in-vector and out-vector if
        # (i) this is the first to compute the vectors or
        # (ii) the dimension is different from that for the previous call of transform function
        if self.out_vec is None:
            self.update_embedding(dim)
        elif self.out_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.out_vec if return_out_vector else self.in_vec

    def update_embedding(self, dim):
        """Update embedding."""
        pass

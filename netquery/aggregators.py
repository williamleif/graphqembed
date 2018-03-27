import torch
import torch.nn as nn
import itertools
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

import random
import math
import numpy as np

"""
Set of modules for aggregating embeddings of neighbors.
These modules take as input embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False): 
        """
        Initializes the aggregator for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        
    def forward(self, to_neighs, rel, keep_prob=0.5, max_keep=10):
        """
        Aggregates embeddings for a batch of nodes.
        keep_prob and max_keep are the parameters for edge/neighbour dropout.

        to_neighs -- list of neighbors of nodes
        keep_prob -- probability of keeping a neighbor
        max_keep  -- maximum number of neighbors kept per node
        """

        # Local pointers to functions (speed hack)
        _int = int
        _set = set
        _min = min
        _len = len
        _ceil = math.ceil
        _sample = random.sample
        samp_neighs = [_set(_sample(to_neigh, 
                        _min(_int(_ceil(_len(to_neigh)*keep_prob)), max_keep)
                        )) for to_neigh in to_neighs]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        embed_matrix = self.features(unique_nodes_list, rel[-1])
        if len(embed_matrix.size()) == 1:
            embed_matrix = embed_matrix.unsqueeze(dim=0)
        to_feats = mask.mm(embed_matrix)
        return to_feats

class FastMeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False): 
        """
        Initializes the aggregator for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        """

        super(FastMeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        
    def forward(self, to_neighs, rel, keep_prob=None, max_keep=25):
        """
        Aggregates embeddings for a batch of nodes.
        keep_prob and max_keep are the parameters for edge/neighbour dropout.

        to_neighs -- list of neighbors of nodes
        keep_prob -- probability of keeping a neighbor
        max_keep  -- maximum number of neighbors kept per node
        """
        _random = random.random
        _int = int 
        _len = len
        samp_neighs = [to_neigh[_int(_random()*_len(to_neigh))] for i in itertools.repeat(None, max_keep) 
                for to_neigh in to_neighs]
        embed_matrix = self.features(samp_neighs, rel[-1])
        to_feats = embed_matrix.view(max_keep, len(to_neighs), embed_matrix.size()[1])
        return to_feats.mean(dim=0)

class PoolAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean pooling of neighbors' embeddings
    """
    def __init__(self, features, feature_dims, cuda=False): 
        """
        Initializes the aggregator for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        """

        super(PoolAggregator, self).__init__()

        self.features = features
        self.feat_dims = feature_dims
        self.pool_matrix = {}
        for mode, feat_dim in self.feat_dims.iteritems():
            self.pool_matrix[mode] = nn.Parameter(torch.FloatTensor(feat_dim, feat_dim))
            init.xavier_uniform(self.pool_matrix[mode])
            self.register_parameter(mode+"_pool", self.pool_matrix[mode])
        self.cuda = cuda
        
    def forward(self, to_neighs, rel, keep_prob=0.5, max_keep=10):
        """
        Aggregates embeddings for a batch of nodes.
        keep_prob and max_keep are the parameters for edge/neighbour dropout.

        to_neighs -- list of neighbors of nodes
        keep_prob -- probability of keeping a neighbor
        max_keep  -- maximum number of neighbors kept per node
        """
        _int = int
        _set = set
        _min = min
        _len = len
        _ceil = math.ceil
        _sample = random.sample
        samp_neighs = [_set(_sample(to_neigh, 
                        _min(_int(_ceil(_len(to_neigh)*keep_prob)), max_keep)
                        )) for to_neigh in to_neighs]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        mode = rel[0]
        if self.cuda:
            mask = mask.cuda()
        embed_matrix = self.features(unique_nodes, rel[-1]).mm(self.pool_matrix[mode])
        to_feats = F.relu(mask.mm(embed_matrix))
        return to_feats

class FastPoolAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean pooling of neighbors' embeddings
    """
    def __init__(self, features, feature_dims,
            cuda=False): 
        """
        Initializes the aggregator for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        """

        super(FastPoolAggregator, self).__init__()

        self.features = features
        self.feat_dims = feature_dims
        self.pool_matrix = {}
        for mode, feat_dim in self.feat_dims.iteritems():
            self.pool_matrix[mode] = nn.Parameter(torch.FloatTensor(feat_dim, feat_dim))
            init.xavier_uniform(self.pool_matrix[mode])
            self.register_parameter(mode+"_pool", self.pool_matrix[mode])
        self.cuda = cuda
        
    def forward(self, to_neighs, rel, keep_prob=0.5, max_keep=10):
        """
        Aggregates embeddings for a batch of nodes.
        keep_prob and max_keep are the parameters for edge/neighbour dropout.

        to_neighs -- list of neighbors of nodes
        keep_prob -- probability of keeping a neighbor
        max_keep  -- maximum number of neighbors kept per node
        """
        _random = random.random
        _int = int 
        _len = len
        samp_neighs = [to_neigh[_int(_random()*_len(to_neigh))] for i in itertools.repeat(None, max_keep) 
                for to_neigh in to_neighs]
        mode = rel[0]
        embed_matrix = self.features(samp_neighs, rel[-1]).mm(self.pool_matrix[mode])
        to_feats = embed_matrix.view(max_keep, len(to_neighs), embed_matrix.size()[1])
        return to_feats.mean(dim=0)

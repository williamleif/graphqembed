import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

"""
Set of modules for encoding nodes.
These modules take as input node ids and output embeddings.
"""

class DirectEncoder(nn.Module):
    """
    Encodes a node as a embedding via direct lookup.
    (i.e., this is just like basic node2vec or matrix factorization)
    """
    def __init__(self, features, feature_modules): 
        """
        Initializes the model for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        feature_modules  -- This should be a map from mode -> torch.nn.EmbeddingBag 
        """
        super(DirectEncoder, self).__init__()
        for name, module in feature_modules.iteritems():
            self.add_module("feat-"+name, module)
        self.features = features

    def forward(self, nodes, mode, offset=None, **kwargs):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        mode      -- string desiginating the mode of the nodes
        offsets   -- specifies how the embeddings are aggregated. 
                     see torch.nn.EmbeddingBag for format. 
                     No aggregation if offsets is None
        """

        if offset is None:
            embeds = self.features(nodes, mode).t()
            norm = embeds.norm(p=2, dim=0, keepdim=True)
            return embeds.div(norm.expand_as(embeds))
        else:
            return self.features(nodes, mode, offset).t()

class Encoder(nn.Module):
    """
    Encodes a node's using a GCN/GraphSage approach
    """
    def __init__(self, features, feature_dims, 
            out_dims, relations, adj_lists, aggregator,
            base_model=None, cuda=False, 
            layer_norm=True,
            feature_modules={}): 
        """
        Initializes the model for a specific graph.

        features         -- function mapping (node_list, features, offset) to feature values
                            see torch.nn.EmbeddingBag and forward function below docs for offset meaning.
        feature_dims     -- output dimension of each of the feature functions. 
        out_dims         -- embedding dimensions for each mode (i.e., output dimensions)
        relations        -- map from mode -> out_going_relations
        adj_lists        -- map from relation_tuple -> node -> list of node's neighbors
        base_model       -- if features are from another encoder, pass it here for training
        cuda             -- whether or not to move params to the GPU
        feature_modules  -- if features come from torch.nn module, pass the modules here for training
        """

        super(Encoder, self).__init__()

        self.features = features
        self.feat_dims = feature_dims
        self.adj_lists = adj_lists
        self.relations = relations
        self.aggregator = aggregator
        for name, module in feature_modules.iteritems():
            self.add_module("feat-"+name, module)
        if base_model != None:
            self.base_model = base_model

        self.out_dims = out_dims
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.layer_norm = layer_norm
        self.compress_dims = {}
        for source_mode in relations:
            self.compress_dims[source_mode] = self.feat_dims[source_mode]
            for (to_mode, _) in relations[source_mode]:
                self.compress_dims[source_mode] += self.feat_dims[to_mode]

        self.self_params = {}
        self.compress_params = {}
        self.lns = {}
        for mode, feat_dim in self.feat_dims.iteritems():
            if self.layer_norm:
                self.lns[mode] = LayerNorm(out_dims[mode])
                self.add_module(mode+"_ln", self.lns[mode])
            self.compress_params[mode] = nn.Parameter(
                    torch.FloatTensor(out_dims[mode], self.compress_dims[mode]))
            init.xavier_uniform(self.compress_params[mode])
            self.register_parameter(mode+"_compress", self.compress_params[mode])

    def forward(self, nodes, mode, keep_prob=0.5, max_keep=10):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        mode      -- string desiginating the mode of the nodes
        """
        self_feat = self.features(nodes, mode).t()
        neigh_feats = []
        for to_r in self.relations[mode]:
            rel = (mode, to_r[1], to_r[0])
            to_neighs = [[-1] if node == -1 else self.adj_lists[rel][node] for node in nodes]
            
            # Special null neighbor for nodes with no edges of this type
            to_neighs = [[-1] if len(l) == 0 else l for l in to_neighs]
            to_feats = self.aggregator.forward(to_neighs, rel, keep_prob, max_keep)
            to_feats = to_feats.t()
            neigh_feats.append(to_feats)
        
        neigh_feats.append(self_feat)
        combined = torch.cat(neigh_feats, dim=0)
        combined = self.compress_params[mode].mm(combined)
        if self.layer_norm:
            combined = self.lns[mode](combined.t()).t()
        combined = F.relu(combined)
        return combined


class LayerNorm(nn.Module):
    """
    Simple layer norm object optionally used with the convolutional encoder.
    """

    def __init__(self, feature_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

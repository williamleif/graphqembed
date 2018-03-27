import torch
import torch.nn as nn 
import numpy as np

import random
from netquery.graph import _reverse_relation

EPS = 10e-6

"""
End-to-end autoencoder models for representation learning on
heteregenous graphs/networks
"""

class MetapathEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons over metapaths
    """

    def __init__(self, graph, enc, dec):
        """
        graph -- simple graph object; see graph.py
        enc --- an encoder module that generates embeddings (see encoders.py) 
        dec --- an decoder module that predicts compositional relationships, i.e. metapaths, between nodes given embeddings. (see decoders.py)
                Note that the decoder must be an *compositional/metapath* decoder (i.e., with name Metapath*.py)
        """
        super(MetapathEncoderDecoder, self).__init__()
        self.enc = enc
        self.dec = dec
        self.graph = graph

    def forward(self, nodes1, nodes2, rels):
        """
        Returns a vector of 'relationship scores' for pairs of nodes being connected by the given metapath (sequence of relations).
        Essentially, the returned scores are the predicted likelihood of the node pairs being connected
        by the given metapath, where the pairs are given by the ordering in nodes1 and nodes2,
        i.e. the first node id in nodes1 is paired with the first node id in nodes2.
        """
        return self.dec.forward(self.enc.forward(nodes1, rels[0][0]), 
                self.enc.forward(nodes2, rels[-1][-1]),
                rels)

    def margin_loss(self, nodes1, nodes2, rels):
        """
        Standard max-margin based loss function.
        Maximizes relationaship scores for true pairs vs negative samples.
        """
        affs = self.forward(nodes1, nodes2, rels)
        neg_nodes = [random.randint(1,len(self.graph.adj_lists[_reverse_relation[rels[-1]]])-1) for _ in xrange(len(nodes1))]
        neg_affs = self.forward(nodes1, neg_nodes,
            rels)
        margin = 1 - (affs - neg_affs)
        margin = torch.clamp(margin, min=0)
        loss = margin.mean()
        return loss 

class QueryEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons about edges, metapaths and intersections
    """

    def __init__(self, graph, enc, edge_dec, inter_dec):
        super(QueryEncoderDecoder, self).__init__()
        self.enc = enc
        self.edge_dec = edge_dec
        self.inter_dec = inter_dec
        self.graph = graph
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, nodes1, nodes2, rels, type_rel="path", target_nodes=[], nodes3=[]):
        if type_rel == "path":
            return self.edge_dec.forward(self.enc.forward(nodes1, rels[0][0]), 
                self.enc.forward(nodes2, rels[-1][-1]), rels)
        elif type_rel == "intersect":
            # get node embeddings of target and anchors
            target_embeds = self.enc(target_nodes, rels[0][-1][-1])
            embeds1 = self.enc(nodes1, rels[0][0][0]) #zi
            embeds2 = self.enc(nodes2, rels[1][0][0]) #zj
            if len(rels)==3:
                embeds3 = self.enc(nodes3, rels[2][0][0])
            
            #Use relationship matrices to get query embeddings
            act1 = embeds1
            for i_rel in rels[0]:
                act1 = self.edge_dec.mats[i_rel].mm(act1)
            query_embeds1 = act1
            
            act2 = embeds2
            for i_rel in rels[1]:
                act2 = self.edge_dec.mats[i_rel].mm(act2)
            query_embeds2 = act2
            
            if len(rels)==3:
                act3 = embeds3
                for i_rel in rels[2]:
                    act3 = self.edge_dec.mats[i_rel].mm(act3)
                    query_embeds3 = act3
               
            #run the intersection operation and final multiplication
            if len(rels) == 3:
                query_intersection = self.inter_dec(query_embeds1, query_embeds2, rels[0][-1][-1], query_embeds3)
                scores = self.cos(target_embeds, query_intersection)
            #    scores = torch.sum(target_embeds*query_intersection, dim=0)
                return scores
            else:
                query_intersection = self.inter_dec(query_embeds1, query_embeds2, rels[0][-1][-1])
                scores = self.cos(target_embeds, query_intersection)
            #    scores = torch.sum(target_embeds*query_intersection, dim=0)
                return scores

    def margin_loss(self, nodes1, nodes2, rels, type_rel="path", neg_nodes=[], target_nodes=[], nodes3 = []):
        if type_rel == "path":
            neg_nodes = [random.randint(1,len(self.graph.adj_lists[_reverse_relation(rels[-1])])-1) for _ in xrange(len(nodes1))]
            neg_nodes = self.graph.adj_lists[_reverse_relation(rels[-1])].keys()
            neg_nodes = np.random.choice(neg_nodes, size=len(nodes1))
            affs = self.forward(nodes1, nodes2, rels)
            neg_affs = self.forward(nodes1, neg_nodes, rels)
            margin = 1 - (affs - neg_affs)
            margin = torch.clamp(margin, min=0)
            loss = margin.mean()
            return loss 
        elif type_rel == "intersect":
            affs = self.forward(nodes1, nodes2, rels, "intersect", target_nodes,nodes3)
            neg_affs = self.forward(nodes1, nodes2, rels, "intersect", neg_nodes,nodes3)
            margin = 1 - (affs - neg_affs)
            margin = torch.clamp(margin, min=0)
            loss = margin.mean()
            return loss

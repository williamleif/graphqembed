import torch
import torch.nn as nn
from torch.nn import init

import numpy as np
import torch.nn.functional as F
"""
A set of decoder modules.
Each decoder takes pairs of embeddings and predicts relationship scores given these embeddings.
"""

""" 
*Edge decoders*
For all edge decoders, the forward method returns a simple relationships score, 
i.e. the likelihood of an edge, between a pair of nodes.
"""

class CosineEdgeDecoder(nn.Module):
    """
    Simple decoder where the relationship score is just the cosine
    similarity between the two embeddings.
    Note: this does not distinguish between edges types
    """

    def __init__(self):
        super(CosineEdgeDecoder, self).__init__()
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, embeds1, embeds2, rel):
        return self.cos(embeds1, embeds2)

class DotProductEdgeDecoder(nn.Module):
    """
    Simple decoder where the relationship score is just the dot product
    between the embeddings (i.e., unnormalized version of cosine)
    Note: this does not distinguish between edges types
    """

    def __init__(self):
        super(DotProductEdgeDecoder, self).__init__()

    def forward(self, embeds1, embeds2, rel):
        dots = torch.sum(embeds1 * embeds2, dim=0)
        return dots

class BilinearEdgeDecoder(nn.Module):
    """
    Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned matrix per relationship type).
    """

    def __init__(self, relations, dims):
        super(BilinearEdgeDecoder, self).__init__()
        self.relations = relations
        self.mats = {}
        self.cos = nn.CosineSimilarity(dim=0)
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.mats[rel] = nn.Parameter(
                        torch.FloatTensor(dims[rel[0]], dims[rel[2]]))
                init.xavier_uniform(self.mats[rel])
                self.register_parameter("_".join(rel), self.mats[rel])

    def forward(self, embeds1, embeds2, rel):
        acts = embeds1.t().mm(self.mats[rel])
        return self.cos(acts.t(), embeds2)

class TransEEdgeDecoder(nn.Module):
    """
    Decoder where the relationship score is given by translation of
    the embeddings (i.e., one learned vector per relationship type).
    """

    def __init__(self, relations, dims):
        super(TransEEdgeDecoder, self).__init__()
        self.relations = relations
        self.vecs = {}
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.vecs[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]]))
                init.uniform(self.vecs[rel], a=-6.0/np.sqrt(dims[rel[0]]), b=6.0/np.sqrt(dims[rel[0]]))
                self.register_parameter("_".join(rel), self.vecs[rel])

    def forward(self, embeds1, embeds2, rel):
        trans_embed = embeds1 + self.vecs[rel].unsqueeze(1).expand(self.vecs[rel].size(0), embeds1.size(1))
        trans_dist = (trans_embed - embeds2).pow(2).sum(0)
        return trans_dist

class BilinearDiagEdgeDecoder(nn.Module):
    """
    Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned diagonal matrix per relationship type).
    """

    def __init__(self, relations, dims):
        super(BilinearDiagEdgeDecoder, self).__init__()
        self.relations = relations
        self.vecs = {}
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.vecs[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]]))
                init.uniform(self.vecs[rel], a=-6.0/np.sqrt(dims[rel[0]]), b=6.0/np.sqrt(dims[rel[0]]))
                self.register_parameter("_".join(rel), self.vecs[rel])

    def forward(self, embeds1, embeds2, rel):
        acts = (embeds1*self.vecs[rel].unsqueeze(1).expand(self.vecs[rel].size(0), embeds1.size(1))*embeds2).sum(0)
        return acts

""" 
*Metapath decoders*
For all metapath encoders, the forward method returns a compositonal relationships score, 
i.e. the likelihood of compositonional relationship or metapath, between a pair of nodes.
"""

class BilinearMetapathDecoder(nn.Module):
    """
    Each edge type is represented by a matrix, and
    compositional relationships are a product matrices.
    """

    def __init__(self, relations, dims):
        super(BilinearMetapathDecoder, self).__init__()
        self.relations = relations
        self.mats = {}
        self.sigmoid = torch.nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=0)
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.mats[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]], dims[rel[2]]))
                init.xavier_uniform(self.mats[rel])
                self.register_parameter("_".join(rel), self.mats[rel])

    def forward(self, embeds1, embeds2, rels):
        act = embeds1.t()
        for i_rel in rels:
            act = act.mm(self.mats[i_rel])
        act = self.cos(act.t(), embeds2)
        return act

class DotBilinearMetapathDecoder(nn.Module):
    """
    Each edge type is represented by a matrix, and
    compositional relationships are a product matrices.
    """

    def __init__(self, relations, dims):
        super(DotBilinearMetapathDecoder, self).__init__()
        self.relations = relations
        self.mats = {}
        self.sigmoid = torch.nn.Sigmoid()
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.mats[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]], dims[rel[2]]))
                #init.xavier_uniform(self.mats[rel])
                init.normal(self.mats[rel], std=0.1)
                self.register_parameter("_".join(rel), self.mats[rel])

    def forward(self, embeds1, embeds2, rels):
        act = embeds1.t()
        for i_rel in rels:
            act = act.mm(self.mats[i_rel])
        dots = torch.sum(act * embeds2, dim=0)
        return dots


class TransEMetapathDecoder(nn.Module):
    """
    Decoder where the relationship score is given by translation of
    the embeddings, each relation type is represented by a vector, and
    compositional relationships are addition of these vectors
    """

    def __init__(self, relations, dims):
        super(TransEMetapathDecoder, self).__init__()
        self.relations = relations
        self.vecs = {}
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.vecs[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]]))
                init.uniform(self.vecs[rel], a=-6.0/np.sqrt(dims[rel[0]]), b=6.0/np.sqrt(dims[rel[0]]))
                self.register_parameter("_".join(rel), self.vecs[rel])

    def forward(self, embeds1, embeds2, rels):
        trans_embed = embeds1
        for i_rel in rels:
            trans_embed += self.vecs[i_rel].unsqueeze(1).expand(self.vecs[i_rel].size(0), embeds1.size(1))
        trans_dist = (trans_embed - embeds2).pow(2).sum(0)
        return trans_dist

class BilinearDiagMetapathDecoder(nn.Module):
    """
    Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned diagonal matrix per relationship type).
    """

    def __init__(self, relations, dims):
        super(BilinearDiagMetapathDecoder, self).__init__()
        self.relations = relations
        self.vecs = {}
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.vecs[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]]))
                init.uniform(self.vecs[rel], a=-6.0/np.sqrt(dims[rel[0]]), b=6.0/np.sqrt(dims[rel[0]]))
                self.register_parameter("_".join(rel), self.vecs[rel])

    def forward(self, embeds1, embeds2, rels):
        acts = embeds1
        for i_rel in rels:
            acts = acts*self.vecs[i_rel].unsqueeze(1).expand(self.vecs[i_rel].size(0), embeds1.size(1))
        acts = (acts*embeds2).sum(0)
        return acts


"""
Set intersection operators. (Experimental)
"""

class TensorIntersection(nn.Module):
    """
    Decoder that computes the implicit intersection between two state vectors
    Uses a symmetric tensor operation.
    """
    def __init__(self, dims):
        super(TensorIntersection, self).__init__()
        self.inter_tensors = {}
        for mode in dims:
            dim = dims[mode]
            self.inter_tensors[mode] = nn.Parameter(torch.FloatTensor(dim, dim, dim))
            init.xavier_uniform(self.inter_tensors[mode])
            self.register_parameter(mode+"_mat", self.inter_tensors[mode])

    def forward(self, embeds1, embeds2, mode):
        inter_tensor = self.inter_tensors[mode] 
        tensor_size = inter_tensor.size()
        inter_tensor = inter_tensor.view(tensor_size[0]*tensor_size[1], tensor_size[2])

        temp1 = inter_tensor.mm(embeds1)
        temp1 = temp1.view(tensor_size[0], tensor_size[1], embeds2.size(1))
        temp2 = inter_tensor.mm(embeds2)
        temp2 = temp2.view(tensor_size[0], tensor_size[1], embeds2.size(1))
        result = (temp1*temp2).sum(dim=1)
        return result

class SetIntersection(nn.Module):
    """
    Decoder that computes the implicit intersection between two state vectors
    Applies an MLP and takes elementwise mins.
    """
    def __init__(self, mode_dims, expand_dims, agg_func=torch.min):
        super(SetIntersection, self).__init__()
        self.pre_mats = {}
        self.post_mats = {}
        self.agg_func = agg_func
        for mode in mode_dims:
            self.pre_mats[mode] = nn.Parameter(torch.FloatTensor(expand_dims[mode], mode_dims[mode]))
            init.xavier_uniform(self.pre_mats[mode])
            self.register_parameter(mode+"_premat", self.pre_mats[mode])
            self.post_mats[mode] = nn.Parameter(torch.FloatTensor(mode_dims[mode], expand_dims[mode]))
            init.xavier_uniform(self.post_mats[mode])
            self.register_parameter(mode+"_postmat", self.post_mats[mode])

    def forward(self, embeds1, embeds2, mode, embeds3 = []):
        temp1 = F.relu(self.pre_mats[mode].mm(embeds1))
        temp2 = F.relu(self.pre_mats[mode].mm(embeds2))
        if len(embeds3) > 0:
            temp3 = F.relu(self.pre_mats[mode].mm(embeds3))
            combined = torch.stack([temp1, temp2, temp3])
        else:
            combined = torch.stack([temp1, temp2])
        combined = self.agg_func(combined,dim=0)
        if type(combined) == tuple:
            combined = combined[0]
        combined = self.post_mats[mode].mm(combined)
        return combined
       
class SimpleSetIntersection(nn.Module):
    """
    Decoder that computes the implicit intersection between two state vectors.
    Takes a simple element-wise min.
    """
    def __init__(self, agg_func=torch.min):
        super(SimpleSetIntersection, self).__init__()
        self.agg_func = agg_func

    def forward(self, embeds1, embeds2, mode, embeds3 = []):
        if len(embeds3) > 0:
            combined = torch.stack([embeds1, embeds2, embeds3])
        else:
            combined = torch.stack([embeds1, embeds2])
        combined = self.agg_func(combined, dim=0)
        if type(combined) == tuple:
            combined = combined[0]
        return combined

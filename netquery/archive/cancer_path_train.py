import time
import random
import numpy as np
import scipy as sp
import torch.optim as optim
import dill as pickle
from sklearn.metrics import roc_auc_score
from argparse import ArgumentParser

from netquery.model import *
from netquery.graph import Graph, _reverse_relation

"""
Code for relationship prediction in multi-modal/hetergenous cancer network.
Experimental fork of cancer_train.py that includes compositonal relationships.
"""

def evaluate_edge_auc(test_edges, graph, enc_dec, batch_size=512):
    predictions = []
    labels = []
    for rel in test_edges:
        print "Testing on", rel
        node_set = set(graph.adj_lists[rel].keys())
        rel_pos_edges = test_edges[rel]
        rel_neg_edges = [(np.random.choice(list(node_set 
            - set(graph.adj_lists[_reverse_relation(rel)][e[1]]))),
            e[1]) for e in rel_pos_edges]
        labels.extend([1 for _ in rel_pos_edges] + [0 for _ in rel_neg_edges])
        edges = rel_pos_edges + rel_neg_edges
        splits = len(edges) / batch_size + 1
        for edge_split in np.array_split(edges, splits):
            scores = enc_dec.forward([e[0] for e in edge_split],
                    [e[1] for e in edge_split], [rel])
            predictions.extend(scores.data.tolist())
    return roc_auc_score(labels, predictions)

def evaluate_edge_mrr(test_edges, graph, enc_dec, negative=100):
    np.random.seed(0)
    mrrs = []
    for i, test_edge in enumerate(test_edges): 
        neg_edges = zip(*[graph.sample_negative_edge(test_edge[2]) for _ in range(negative)])
        scores = enc_dec.forward([test_edge[0]]+list(neg_edges[0]), 
                [test_edge[1]]+list(neg_edges[1]), [test_edge[2]]).data.cpu().numpy()
        mrrs.append(1./sp.stats.rankdata(-1*scores)[0])
    return np.mean(mrrs)

def evaluate_edge_margin(test_edges, graph, enc_dec, negative=100):
    np.random.seed(0)
    loss = 0.
    for i, test_edge in enumerate(test_edges): 
        loss += enc_dec.margin_loss([test_edge[0] for _ in range(negative)],
                [test_edge[1] for _ in range(negative)], [test_edge[2]]).data[0]
    loss /= len(test_edges)
    return loss
'''
def train(feature_dim, lr, model, batch_size, max_batches, tol):
    relations, adj_lists, node_maps = pickle.load(open("/dfs/scratch0/netquery/cancer.pkl"))
    relations['disease'].remove(('disease', '0'))
    del adj_lists[('disease', '0', 'disease')]

    for mode in node_maps:
        node_maps[mode][-1] = len(node_maps[mode])
    feature_dims = {mode : feature_dim for mode in relations}
    feature_modules = {mode : nn.EmbeddingBag(len(node_maps[mode]), 
        feature_dim) for mode in relations}
    cuda = True
    if cuda:
        features = lambda nodes, mode, offset : feature_modules[mode].forward(
                Variable(torch.LongTensor(
                    [node_maps[mode][node] for node in nodes])).cuda(), Variable(torch.LongTensor(offset)).cuda())
    else:
        features = lambda nodes, mode, offset : feature_modules[mode].forward(
                Variable(torch.LongTensor(
                    [node_maps[mode][node] for node in nodes])), Variable(torch.LongTensor(offset)))
    for feature_module in feature_modules.values():
        feature_module.weight.data.normal_(0, 1./np.sqrt(feature_dim))

    graph = Graph(features, feature_dims, relations, adj_lists)
    edges = graph.get_all_edges_byrel()
    train_edges = {rel:edge_list[:int(0.9*len(edge_list))] for rel, edge_list in edges.iteritems()}
    test_edges = {rel:edge_list[int(0.9*len(edge_list)):] for rel, edge_list in edges.iteritems()}
    graph.remove_edges([e for edge_list in test_edges.values() for e in edge_list])

    out_dims = {mode:feature_dim for mode in graph.relations}
    if model == "direct":
        direct_enc = DirectEncoder(graph.features, feature_modules)
        dec = BilinearPathDecoder(graph.relations, graph.feature_dims)
        enc_dec = PathEncoderDecoder(graph, direct_enc, dec)
    else:
        enc1 = Encoder(graph.features, 
            graph.feature_dims, 
            out_dims, 
            graph.relations, 
            graph.adj_lists, concat=True, feature_modules=feature_modules,
            cuda = cuda)
        dec = BilinearPathDecoder(graph.relations, enc1.out_dims)
        enc_dec = PathEncoderDecoder(graph, enc1, dec)
    if cuda:
        enc_dec.cuda()
    optimizer = optim.SGD(enc_dec.parameters(), lr=lr, momentum=0.000)

    start = time.time()
    ema_loss = None
    print "{:d} training edges".format(sum([len(rel_edges) for rel_edges in train_edges.values()]))
    for k in range(max_path_len):
        losses = []
        print "Starting training for metapaths of length", k+1
        for i in range(max_batches):
            if k == 0:
                rel = graph.sample_relation()
                random.shuffle(train_edges[rel])
                edges = train_edges[rel][:batch_size]
                if len(edges) == 0:
                    continue
                optimizer.zero_grad()
                graph.remove_edges(edges)
                loss = enc_dec.margin_loss([edge[0] for edge in edges], 
                        [edge[1] for edge in edges], rel)
                graph.add_edges(edges)
            else:
                rels = graph.sample_metapath()
                nodes1, nodes2 = zip(*[graph.sample_path_with_rels(rels) for _ in range(batch_size)])
                
                optimizer.zero_grad()
                loss = enc_dec.margin_loss(nodes1, 
                        nodes2, rels)
            losses.append(loss.data[0])
            if ema_loss == None:
                ema_loss = loss.data[0]
            else:
                ema_loss = 0.99*ema_loss + 0.01*loss.data[0]
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print i, ema_loss
            if i > 2000 and i % 100 == 0:
                conv = np.mean(losses[i-2000:i-1000]) - np.mean(losses[i-1000:i]) 
                print "conv", conv
                if conv < tol:
                    break
        print "Stats after paths of length", k+1
        total = time.time() - start
        print "Time:", total
        print "Converged after:", i
        print "Per example:", total/batch_size/float(i)
        print "AUC:", evaluate_edge_auc(test_edges, graph, enc_dec)
'''
def train(feature_dim, lr, model, batch_size, max_batches, tol, max_path_len):
    feature_dim = 16
    relations, adj_lists, node_maps = pickle.load(open("/dfs/scratch0/netquery/cancer.pkl"))
    relations['disease'].remove(('disease', '0'))
    del adj_lists[('disease', '0', 'disease')]
    for rel1 in relations:
        for rel2  in relations[rel1]:
            print rel1, rel2, len(adj_lists[(rel1, rel2[1], rel2[0])])
    for mode in node_maps:
        node_maps[mode][-1] = len(node_maps[mode])
    feature_dims = {mode : feature_dim for mode in relations}
    feature_modules = {mode : nn.EmbeddingBag(len(node_maps[mode]), 
        feature_dim) for mode in relations}
    for feature_module in feature_modules.values():
        feature_module.weight.data.normal_(0, 1./np.sqrt(feature_dim))
    cuda = True
    if cuda:
        features = lambda nodes, mode, offset : feature_modules[mode].forward(
                Variable(torch.LongTensor(
                    [node_maps[mode][node] for node in nodes])).cuda(), 
                Variable(torch.LongTensor(offset)).cuda())
    else:
        features = lambda nodes, mode, offset : feature_modules[mode].forward(
                Variable(torch.LongTensor(
                    [node_maps[mode][node] for node in nodes])), 
                Variable(torch.LongTensor(offset)))

    graph = Graph(features, feature_dims, relations, adj_lists)
    edges = graph.get_all_edges_byrel()
    train_edges = {rel:edge_list[:int(0.9*len(edge_list))] for rel, edge_list in edges.iteritems()}
    test_edges = {rel:edge_list[int(0.9*len(edge_list)):] for rel, edge_list in edges.iteritems()}
    graph.remove_edges([e for edge_list in test_edges.values() for e in edge_list])

    direct_enc = DirectEncoder(graph.features, feature_modules)
    dec = BilinearPathDecoder(graph.relations, feature_dims)
    enc_dec = PathEncoderDecoder(graph, direct_enc, dec)
    if cuda:
        enc_dec.cuda()
    optimizer = optim.SGD(enc_dec.parameters(), lr=0.5, momentum=0.000)

    start = time.time()
    print "{:d} training edges".format(sum([len(rel_edges) for rel_edges in train_edges.values()]))
    batch_size = 512
    num_batches = 20000
    tol = 0.0001
    losses = []
    ema_loss = None
    for i in range(num_batches):
        rel = graph.sample_relation()
        random.shuffle(train_edges[rel])
        edges = train_edges[rel][:batch_size]
        if len(edges) == 0:
            continue
        optimizer.zero_grad()
        enc_dec.graph.remove_edges(edges)
        loss = enc_dec.margin_loss([edge[0] for edge in edges], 
                [edge[1] for edge in edges], [rel])
        enc_dec.graph.add_edges(edges)
        losses.append(loss.data[0])
        if ema_loss == None:
            ema_loss = loss.data[0]
        else:
            ema_loss = 0.99*ema_loss + 0.01*loss.data[0]
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print i, ema_loss
        if i > 2000 and i % 100 == 0:
            conv = np.mean(losses[i-2000:i-1000]) - np.mean(losses[i-1000:i]) 
            print "conv", conv
            if conv < tol:
                break
    print "MRR:", evaluate_edge_auc(test_edges, graph, enc_dec)

    batch_size = 512
    num_batches = 100000
    ema_loss = None
    optimizer = optim.SGD(enc_dec.parameters(), lr=0.5, momentum=0.000)
    for i in range(num_batches):
        rels = graph.sample_metapath()
        nodes1, nodes2 = zip(*[graph.sample_path_with_rels(rels) for _ in range(batch_size)])
        
        optimizer.zero_grad()
        loss = enc_dec.margin_loss(nodes1, 
                nodes2, rels)
        losses.append(loss.data[0])
        if ema_loss == None:
            ema_loss = loss.data[0]
        else:
            ema_loss = 0.99*ema_loss + 0.01*loss.data[0]
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print i, ema_loss
        if i % 5000 == 0:
            print "MRR:", evaluate_edge_auc(test_edges, graph, enc_dec)
 
    total = time.time() - start
    print "Time:", total
    print "Converged after:", i
    print "Per example:", total/batch_size/float(i)
    print "MRR:", evaluate_edge_auc(test_edges, graph, enc_dec)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="direct")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_batches", type=int, default=100000)
    parser.add_argument("--max_path_len", type=int, default=2)
    parser.add_argument("--tol", type=float, default=0.0001)
    args = parser.parse_args()
    train(args.feature_dim, args.lr, args.model, args.batch_size, args.max_batches, args.tol, args.max_path_len)

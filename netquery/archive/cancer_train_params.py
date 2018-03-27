import time
import random
import numpy as np
import csv
import torch.optim as optim
import dill as pickle
from sklearn.metrics import roc_auc_score
from argparse import ArgumentParser

from netquery.model import *
from netquery.encoders import *
from netquery.decoders import *
from netquery.aggregators import *
from netquery.graph import Graph, _reverse_relation

"""
Code for relationship prediction in multi-modal/hetergenous cancer network.
"""

def evaluate_edge_margin(test_edges, graph, enc_dec, negative=100):
    """
    Evaluates the margin loss on held out data.
    negative=# of negative samples to use.
    """
    test_edges = [e for sub_list in test_edges.values() for e in sub_list]
    np.random.seed(0)
    loss = 0.
    for i, test_edge in enumerate(test_edges):
        loss += enc_dec.margin_loss([test_edge[0] for _ in range(negative)],
                [test_edge[1] for _ in range(negative)], test_edge[2]).data[0]
    loss /= len(test_edges)
    return loss

def evaluate_edge_auc(test_edges, graph, enc_dec, batch_size=512):
    """
    Evaluates the AUC score for ranking true relationships vs negative samples.
    """
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
                    [e[1] for e in edge_split], rel)
            predictions.extend(scores.data.tolist())
    return roc_auc_score(labels, predictions)

def get_decoder(graph, out_dims, decoder):
    if decoder == "bilinear":
        dec = BilinearEdgeDecoder(graph.relations, out_dims)
    elif decoder == "transe":
        dec = TransEEdgeDecoder(graph.relations, out_dims)
    elif decoder == "bilinear-diag":
        dec = BilinearDiagEdgeDecoder(graph.relations, out_dims)
    elif decoder == "cosine":
        dec = CosineEdgeDecoder()
    elif decoder == "dot-product":
        dec = DotProductEdgeDecoder()
    return dec

def train(feature_dim, lr, model, batch_size, max_batches, tol, cuda, results, decoder, opt, agg):

    # load the data
    # relations, adj_lists, node_maps = pickle.load(open("/dfs/scratch0/netquery/cancer.pkl"))
    relations, adj_lists, node_maps = pickle.load(open("cancer.pkl"))

    # delete this relation because it doesn't have enough data
    relations['disease'].remove(('disease', '0'))
    del adj_lists[('disease', '0', 'disease')]

    # add dummy node (messy hack for nw)
    for mode in node_maps:
        node_maps[mode][-1] = len(node_maps[mode])

    # set the feature dimensions to be equal for all modes
    feature_dims = {mode : feature_dim for mode in relations}
    # the feature modules for all nodes are embedding lookups.
    feature_modules = {mode: nn.Embedding(len(node_maps[mode]), feature_dim) for mode in relations}

    # need to define the feature function that maps nodes to features
    if cuda:
        features = lambda nodes, mode: feature_modules[mode].forward(
                Variable(torch.LongTensor([node_maps[mode][node] for node in nodes])).cuda())
    else:
        features = lambda nodes, mode: feature_modules[mode].forward(
                Variable(torch.LongTensor([node_maps[mode][node] for node in nodes])))
        
    # give reasonable initialization to features
    for feature_module in feature_modules.values():
        feature_module.weight.data.normal_(0, 1./np.sqrt(feature_dim))

    # build the graph
    graph = Graph(features, feature_dims, relations, adj_lists)

    # get mapping from relations->list of edges for that relation
    edges = graph.get_all_edges_byrel()

    # seperate into train and test sets
    train_edges = {rel:edge_list[:int(0.9*len(edge_list))] for rel, edge_list in edges.iteritems()}
    test_edges = {rel:edge_list[int(0.9*len(edge_list)):] for rel, edge_list in edges.iteritems()}
    graph.remove_edges([e for edge_list in test_edges.values() for e in edge_list])

    # for simplicity the embedding and hidden dimensions are equal
    out_dims = {mode:feature_dim for mode in graph.relations}

    # define the encoder.
    # Either direct or based on single-step convolution
    if model == "direct":
        enc = DirectEncoder(graph.features, feature_modules)
        dec = get_decoder(graph, feature_dims, decoder)
        enc_dec = EdgeEncoderDecoder(graph, enc, dec)
    else:
        if agg == "mean":
            aggregator = MeanAggregator(graph.features)
        elif agg == "pool":
            aggregator = PoolAggregator(graph.features, graph.feature_dims)
        enc = Encoder(graph.features, 
            graph.feature_dims, 
            out_dims, 
            graph.relations, 
            graph.adj_lists, concat=True, feature_modules=feature_modules,
            cuda = cuda, aggregator=aggregator)
        dec = get_decoder(graph, enc.out_dims, decoder)
        enc_dec = EdgeEncoderDecoder(graph, enc, dec)
    if cuda:
        enc_dec.cuda()
        
    if opt == "sgd":
        optimizer = optim.SGD(enc_dec.parameters(), lr=lr, momentum=0.000)
    elif opt == "sgd-momentum":
        optimizer = optim.SGD(enc_dec.parameters(), lr=lr, momentum=0.9)
    elif opt == "adam":
        optimizer = optim.Adam(enc_dec.parameters(), lr=lr)

    # Main training loop
    start = time.time()
    ema_loss = None
    print "{:d} training edges".format(sum([len(rel_edges) for rel_edges in train_edges.values()]))
    losses = []
    for i in range(max_batches):
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

    total = time.time() - start
    test_auc = evaluate_edge_auc(test_edges, graph, enc_dec)
    test_loss = evaluate_edge_margin(test_edges, graph, enc_dec)
    with open (results, "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([str(lr), str(batch_size), str(total), str(i), str( total/batch_size/float(i)), str(test_auc), str(test_loss), str(ema_loss), str(conv)]) 
    print "Time:", total
    print "Converged after:", i
    print "Per example:", total/batch_size/float(i)
    print "AUC:", test_auc
    print "Loss:", test_loss

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="direct")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_batches", type=int, default=100000)
    parser.add_argument("--tol", type=float, default=0.0001)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--results", type=str, default="results.csv")
    parser.add_argument("--decoder", type=str, default="bilinear")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--aggregator", type=str, default="mean")
    args = parser.parse_args()
    train(args.feature_dim, args.lr, args.model, args.batch_size, args.max_batches, args.tol, args.cuda, args.results, args.decoder, args.optimizer, args.aggregator)

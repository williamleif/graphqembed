import csv
import time
import random
import numpy as np
import scipy as sp
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
Experimental fork of cancer_train.py that includes compositonal relationships.
"""

def evaluate_metapath_auc(test_metapaths, graph, enc_dec, batch_size=512):
    predictions = []
    labels = []
    for rels in test_metapaths:
        print "Testing on", rels
        if rels[0] == rels[1]:
            continue
        rels_pos_metapaths = test_metapaths[rels]
        if len(rels_pos_metapaths)>0:
            node_set = set(graph.adj_lists[_reverse_relation(rels[1])].keys())
            rels_all_metapaths = graph.get_metapath_byrels(rels)
            rels_neg_metapaths = []
            for e in rels_pos_metapaths:
                sample_space = list(node_set - set(rels_all_metapaths[e[0]]))
                if len(sample_space)>0:
                    rels_neg_metapaths.append((e[0], np.random.choice(sample_space)))

            labels.extend([1 for _ in rels_pos_metapaths] + [0 for _ in rels_neg_metapaths])
            metapaths = rels_pos_metapaths + rels_neg_metapaths
            splits = len(metapaths) / batch_size + 1
            for metapath_split in np.array_split(metapaths, splits):
                scores = enc_dec.forward([e[0] for e in metapath_split],
                        [e[1] for e in metapath_split], rels)
                predictions.extend(scores.data.tolist())
    return roc_auc_score(labels, predictions)

def evaluate_metapath_margin(test_metapaths, graph, enc_dec, negative=100, batch_size=512):
	np.random.seed(0)
	loss = 0.
	for rel in test_metapaths:
		if len(test_metapaths[rel]) > 0:
			rel_pos_edges = test_metapaths[rel]
			edges = [(e[0], e[1]) for e in rel_pos_edges]
			splits = len(edges) / batch_size + 1
			for edge_split in np.array_split(edges, splits):
				loss += enc_dec.margin_loss([e[0] for e in edge_split  for _ in range(negative)],
		            [e[1] for e in edge_split for _ in range(negative)], rel).data[0]*len(edge_split)
	len_test_edges = sum([len(test_metapaths[rel]) for rel in test_metapaths])
	loss /= len_test_edges  
	return loss

def evaluate_edge_auc(test_edges, graph, enc_dec, batch_size=512):
    predictions = []
    labels = []
    for rel in test_edges:
        print "Testing on", rel
        node_set = set(graph.adj_lists[_reverse_relation(rel)].keys())
        rel_pos_edges = test_edges[rel]
        rel_neg_edges = [(e[0],np.random.choice(list(node_set 
            - set(graph.adj_lists[rel][e[0]])))) for e in rel_pos_edges]
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

def evaluate_edge_margin(test_edges, graph, enc_dec, negative=100, batch_size=512):
	np.random.seed(0)
	loss = 0.
	for rel in test_edges:
		rel_pos_edges = test_edges[rel]
		edges = [(e[0], e[1]) for e in rel_pos_edges]
		splits = len(edges) / batch_size + 1
		for edge_split in np.array_split(edges, splits):
			loss += enc_dec.margin_loss([e[0] for e in edge_split  for _ in range(negative)],
	            [e[1] for e in edge_split for _ in range(negative)], [rel]).data[0]*len(edge_split)
	len_test_edges = sum([len(test_edges[rel]) for rel in test_edges])
	loss /= len_test_edges
	return loss

def get_decoder(graph, out_dims, decoder):
    if decoder == "bilinear":
        dec = BilinearMetapathDecoder(graph.relations, out_dims)
    elif decoder == "transe":
        dec = TransEMetapathDecoder(graph.relations, out_dims)
    elif decoder == "bilinear-diag":
        dec = BilinearDiagMetapathDecoder(graph.relations, out_dims)
    return dec

def train(feature_dim, lr, model, batch_size, max_batches, tol, max_path_len, cuda, results, decoder, opt, agg):
    feature_dim = 16
    # relations, adj_lists, node_maps = pickle.load(open("/dfs/scratch0/netquery/cancer.pkl"))
    relations, adj_lists, node_maps = pickle.load(open("cancer.pkl"))
    relations['disease'].remove(('disease', '0'))
    del adj_lists[('disease', '0', 'disease')]
    for rel1 in relations:
        for rel2  in relations[rel1]:
            print rel1, rel2, len(adj_lists[(rel1, rel2[1], rel2[0])])
    for mode in node_maps:
        node_maps[mode][-1] = len(node_maps[mode])
    feature_dims = {mode : feature_dim for mode in relations}
    feature_modules = {mode : nn.Embedding(len(node_maps[mode]), 
        feature_dim) for mode in relations}
    for feature_module in feature_modules.values():
        feature_module.weight.data.normal_(0, 1./np.sqrt(feature_dim))

    if cuda:
        features = lambda nodes, mode: feature_modules[mode].forward(
                Variable(torch.LongTensor([node_maps[mode][node] for node in nodes])).cuda())
    else:
        features = lambda nodes, mode: feature_modules[mode].forward(
                Variable(torch.LongTensor([node_maps[mode][node] for node in nodes])))

    graph = Graph(features, feature_dims, relations, adj_lists)

	# cancer_chains = graph.create_chains_byrels()
    # cancer_pos_ints, cancer_neg_ints = graph.create_intersections_byrels(cancer_chains)

    metapaths = graph.get_all_metapaths_byrel()
    train_metapaths = {rel:metapath_list[:int(0.9*len(metapath_list))] for rel, metapath_list in metapaths.iteritems()}
    test_metapaths = {rel:metapath_list[int(0.9*len(metapath_list)):] for rel, metapath_list in metapaths.iteritems()}

    
    edges = graph.get_all_edges_byrel()
    train_edges = {rel:edge_list[:int(0.9*len(edge_list))] for rel, edge_list in edges.iteritems()}
    test_edges = {rel:edge_list[int(0.9*len(edge_list)):] for rel, edge_list in edges.iteritems()}
    graph.remove_edges([e for edge_list in test_edges.values() for e in edge_list])

    # for simplicity the embedding and hidden dimensions are equal
    out_dims = {mode:feature_dim for mode in graph.relations}

    if model == "direct":
        enc = DirectEncoder(graph.features, feature_modules)
        dec = get_decoder(graph, feature_dims, decoder)
    else:
        if agg == "mean":
            aggregator = FastMeanAggregator(graph.features)
        elif agg == "pool":
            aggregator = FastPoolAggregator(graph.features, graph.feature_dims)
        enc = Encoder(graph.features, 
            graph.feature_dims, 
            out_dims, 
            graph.relations, 
            graph.adj_lists, concat=True, feature_modules=feature_modules,
            cuda = cuda, aggregator=aggregator)
        dec = get_decoder(graph, enc.out_dims, decoder)
    
    enc_dec = MetapathEncoderDecoder(graph, enc, dec)
    if cuda:
        enc_dec.cuda()

    if opt == "sgd":
        optimizer = optim.SGD(enc_dec.parameters(), lr=lr, momentum=0.000)
    elif opt == "sgd-momentum":
        optimizer = optim.SGD(enc_dec.parameters(), lr=lr, momentum=0.9)
    elif opt == "adam":
        optimizer = optim.Adam(enc_dec.parameters(), lr=lr)

    start = time.time()
    print "{:d} training edges".format(sum([len(rel_edges) for rel_edges in train_edges.values()]))
    losses = []
    ema_loss = None
    
    conv = -1
    for i in range(max_batches):
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
    old_edge_auc = evaluate_edge_auc(test_edges, graph, enc_dec)
    print "MRR:", old_edge_auc
    old_edge_loss = evaluate_edge_margin(test_edges, graph, enc_dec)

    old_path_loss = evaluate_metapath_margin(test_metapaths, graph, enc_dec)
    old_path_auc = evaluate_metapath_auc(test_metapaths, graph, enc_dec, batch_size=batch_size)

    print "Metapath auc:", old_path_auc
    print "Metapath margin: ", old_path_loss

    ema_loss = None
    if opt == "sgd":
        optimizer = optim.SGD(enc_dec.parameters(), lr=lr, momentum=0.000)
    elif opt == "sgd-momentum":
        optimizer = optim.SGD(enc_dec.parameters(), lr=lr, momentum=0.9)
    elif opt == "adam":
        optimizer = optim.Adam(enc_dec.parameters(), lr=lr)
    for i in range(max_batches):
        while True:
            rels = graph.sample_metapath()
            random.shuffle(train_metapaths[rels])
            edges = train_metapaths[rels][:batch_size]
            if len(edges) > 0:
                break
        nodes1 = [edge[0] for edge in edges]
        nodes2 = [edge[1] for edge in edges]
        
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
        if i % 5000 == 0:
            print "MRR:", evaluate_edge_auc(test_edges, graph, enc_dec)

    total = time.time() - start
    test_auc = evaluate_edge_auc(test_edges, graph, enc_dec)
    test_loss = evaluate_edge_margin(test_edges, graph, enc_dec)
    path_auc = evaluate_metapath_auc(test_metapaths, graph, enc_dec, batch_size=batch_size)
    path_loss = evaluate_metapath_margin(test_metapaths, graph, enc_dec)

    with open (results, "a") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([str(lr), str(batch_size), str(total), str(i), str(total/batch_size/float(i)), str(old_path_auc), str(old_edge_loss), str(test_auc), str(test_loss), str(ema_loss), str(conv), str(old_path_loss), str(old_path_auc), str(path_loss), str(path_auc)]) 
    
    print "Time:", total
    print "Converged after:", i
    print "Per example:", total/batch_size/float(i)
    print "MRR:", test_auc
    print "Loss:", test_loss
    print "Metapath auc:", path_auc
    print "Metapath margin: ", path_loss

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="direct")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_batches", type=int, default=100000)
    parser.add_argument("--max_path_len", type=int, default=2)
    parser.add_argument("--tol", type=float, default=0.0001)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--results", type=str, default="results.csv")
    parser.add_argument("--decoder", type=str, default="bilinear")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--aggregator", type=str, default="mean")
    args = parser.parse_args()
    train(args.feature_dim, args.lr, args.model, args.batch_size, args.max_batches, args.tol, args.max_path_len, args.cuda, args.results, args.decoder, args.optimizer, args.aggregator)

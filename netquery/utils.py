import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from netquery.decoders import BilinearMetapathDecoder, TransEMetapathDecoder, BilinearDiagMetapathDecoder, SetIntersection, SimpleSetIntersection
from netquery.encoders import DirectEncoder, Encoder
from netquery.aggregators import MeanAggregator
import cPickle as pickle
import logging

"""
Misc utility functions..
"""

def get_val(data, val_prop):
    train_data = []
    val_data = []
    for kind in range(len(data)):
        train_data_kind = {}
        val_data_kind = {}
        for length in data[kind].keys():
            train_data_kind[length] = {rel : [e for e in es[int(len(es)*val_prop):]] for rel, es in data[length].iteritems()}
            val_data_kind[length] = {rel : [e for e in es[:int(len(es)*val_prop)]] for rel, es in data[length].iteritems()}
        train_data.append(train_data_kind)
        val_data.append(val_data_kind)
    return train_data, val_data


def cudify(feature_modules):
   features = lambda nodes, mode : feature_modules[mode](
            torch.autograd.Variable(torch.LongTensor(nodes)+1).cuda())
   return features


def load_train_test_data(data_dir):
    train_chains = {}
    test_chains = {}
    for i in range(1,4):
        train_chains[i] = pickle.load(open(data_dir + "/train_chains_{:d}.pkl".format(i), "rb"))
        test_chains[i] = pickle.load(open(data_dir + "/test_chains_{:d}.pkl".format(i), "rb"))
    train_polys = {}
    test_polys = {}
    for i in range(2,4):
        train_polys[i] = pickle.load(open(data_dir + "/train_polys_{:d}.pkl".format(i), "rb"))
        test_polys[i] = pickle.load(open(data_dir + "/test_polys_{:d}.pkl".format(i), "rb"))
    return train_chains, test_chains, train_polys, test_polys


def evaluate_path_auc(test_paths, neg_paths, enc_dec, batch_size=512):
    """
    Evaluates the AUC score for ranking true relationships vs negative samples.
    """
    predictions = []
    labels = []
    rel_aucs = {}
    for rels in test_paths:
        rel_predictions = []
        rel_labels = []
        rel_pos_edges = test_paths[rels]
        rel_neg_edges = neg_paths[rels]
        rel_labels.extend([1 for _ in rel_pos_edges] + [0 for _ in rel_neg_edges])
        labels.extend([1 for _ in rel_pos_edges] + [0 for _ in rel_neg_edges])
        edges = rel_pos_edges + rel_neg_edges
        if (len(edges)) == 0:
            continue
        splits = len(edges) / batch_size + 1
        for edge_split in np.array_split(edges, splits):
            if type(rels[0]) == str:
                rels = (rels,)
            scores = enc_dec.forward([e[0] for e in edge_split],
                    [e[1] for e in edge_split], rels)
            predictions.extend(scores.data.tolist())
            rel_predictions.extend(scores.data.tolist())
        rel_aucs[rels] =  roc_auc_score(rel_labels, np.nan_to_num(rel_predictions))
    return roc_auc_score(labels, np.nan_to_num(predictions)), rel_aucs

def safe_split(data, num_splits):
    split_size = len(data)/num_splits
    splits = []
    for i in range(num_splits):
        splits.append(data[:min(split_size, len(data))])
        data = data[min(split_size, len(data)):]
    splits = splits + [data] if len(data) > 0 else splits
    return splits 

def evaluate_intersect_auc(test_intersects, neg_intersects, enc_dec, batch_size=512):
    predictions = []
    labels = []
    rel_aucs = {}
    for rels in test_intersects:    
        rels_pos_ints = test_intersects[rels]
        rels_neg_ints = neg_intersects[rels]
        rel_labels = [1 for _ in rels_pos_ints] + [0 for _ in rels_neg_ints]
        labels.extend(rel_labels)
        rel_scores = []
        if len(rels_pos_ints)>0:
            ints = rels_pos_ints + rels_neg_ints
            splits = len(ints) / batch_size + 1
            if len(rels) == 3:
                for int_split in safe_split(ints, splits):
                    scores = enc_dec.forward([e[1][0] for e in int_split],
                        [e[1][1] for e in int_split], rels, 
                        "intersect", 
                        nodes3 = [e[1][2] for e in int_split],
                        target_nodes=[e[0] for e in int_split])
            else:
                for int_split in safe_split(ints, splits):
                    scores = enc_dec.forward([e[1][0] for e in int_split],
                        [e[1][1] for e in int_split], rels, 
                        "intersect", 
                        target_nodes=[e[0] for e in int_split])
                    rel_scores.extend(scores.data.tolist())
            scores =  np.nan_to_num(rel_scores)
            rel_aucs[rels] = roc_auc_score(rel_labels, scores)
            predictions.extend(scores)
    return roc_auc_score(labels, predictions), rel_aucs

def get_encoder(depth, graph, out_dims, feature_modules, cuda): 
    if depth < 0 or depth > 3:
        raise Exception("Depth must be between 0 and 3 (inclusive)")

    if depth == 0:
         enc = DirectEncoder(graph.features, feature_modules)
    else:
        aggregator1 = MeanAggregator(graph.features)
        enc1 = Encoder(graph.features, 
                graph.feature_dims, 
                out_dims, 
                graph.relations, 
                graph.adj_lists, feature_modules=feature_modules, 
                cuda=cuda, aggregator=aggregator1)
        enc = enc1
        if depth >= 2:
            aggregator2 = MeanAggregator(lambda nodes, mode : enc1(nodes, mode).t().squeeze())
            enc2 = Encoder(lambda nodes, mode : enc1(nodes, mode).t().squeeze(),
                    enc1.out_dims, 
                    out_dims, 
                    graph.relations, 
                    graph.adj_lists, base_model=enc1,
                    cuda=cuda, aggregator=aggregator2)
            enc = enc2
            if depth >= 3:
                aggregator3 = MeanAggregator(lambda nodes, mode : enc2(nodes, mode).t().squeeze())
                enc3 = Encoder(lambda nodes, mode : enc1(nodes, mode).t().squeeze(),
                        enc2.out_dims, 
                        out_dims, 
                        graph.relations, 
                        graph.adj_lists, base_model=enc2,
                        cuda=cuda, aggregator=aggregator3)
                enc = enc3
    return enc

def get_metapath_decoder(graph, out_dims, decoder):
    if decoder == "bilinear":
        dec = BilinearMetapathDecoder(graph.relations, out_dims)
    elif decoder == "transe":
        dec = TransEMetapathDecoder(graph.relations, out_dims)
    elif decoder == "bilinear-diag":
        dec = BilinearDiagMetapathDecoder(graph.relations, out_dims)
    else:
        raise Exception("Metapath decoder not recognized.")
    return dec

def get_intersection_decoder(graph, out_dims, decoder):
    if decoder == "mean":
        dec = SetIntersection(out_dims, out_dims, agg_func=torch.mean)
    elif decoder == "mean-simple":
        dec = SimpleSetIntersection(agg_func=torch.mean)
    elif decoder == "max":
        dec = SetIntersection(out_dims, out_dims, agg_func=torch.max)
    elif decoder == "max-simple":
        dec = SimpleSetIntersection(agg_func=torch.max)
    else:
        raise Exception("Intersection decoder not recognized.")
    return dec

def setup_logging(log_file, console=True):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='logs_file',
                        filemode='w')
    if console:
        console = logging.StreamHandler()
        # optional, set the logging level
        console.setLevel(logging.INFO)
        # set a format which is the same for console use
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    return logging

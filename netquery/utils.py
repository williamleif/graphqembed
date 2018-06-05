import numpy as np
import scipy
import scipy.stats as stats
import torch
from sklearn.metrics import roc_auc_score
from netquery.decoders import BilinearMetapathDecoder, TransEMetapathDecoder, BilinearDiagMetapathDecoder, SetIntersection, SimpleSetIntersection
from netquery.encoders import DirectEncoder, Encoder
from netquery.aggregators import MeanAggregator
import cPickle as pickle
import logging
import random

"""
Misc utility functions..
"""

def cudify(feature_modules, node_maps=None):
   if node_maps is None:
       features = lambda nodes, mode : feature_modules[mode](
            torch.autograd.Variable(torch.LongTensor(nodes)+1).cuda())
   else:
       features = lambda nodes, mode : feature_modules[mode](
            torch.autograd.Variable(torch.LongTensor([node_maps[mode][n] for n in nodes])+1).cuda())
   return features

def _get_perc_scores(scores, lengths):
    perc_scores = []
    cum_sum = 0
    neg_scores = scores[len(lengths):]
    for i, length in enumerate(lengths):
        perc_scores.append(stats.percentileofscore(neg_scores[cum_sum:cum_sum+length], scores[i]))
        cum_sum += length
    return perc_scores

def eval_auc_queries(test_queries, enc_dec, batch_size=1000, hard_negatives=False, seed=0):
    predictions = []
    labels = []
    formula_aucs = {}
    random.seed(seed)
    for formula in test_queries:
        formula_labels = []
        formula_predictions = []
        formula_queries = test_queries[formula]
        offset = 0
        while offset < len(formula_queries):
            max_index = min(offset+batch_size, len(formula_queries))
            batch_queries = formula_queries[offset:max_index]
            if hard_negatives:
                lengths = [1 for j in range(offset, max_index)]
                negatives = [random.choice(formula_queries[j].hard_neg_samples) for j in xrange(offset, max_index)]
            else:
                lengths = [1 for j in range(offset, max_index)]
                negatives = [random.choice(formula_queries[j].neg_samples) for j  in xrange(offset, max_index)]
            offset += batch_size

            formula_labels.extend([1 for _ in xrange(len(lengths))])
            formula_labels.extend([0 for _ in xrange(len(negatives))])
            batch_scores = enc_dec.forward(formula, 
                    batch_queries+[b for i, b in enumerate(batch_queries) for _ in range(lengths[i])], 
                    [q.target_node for q in batch_queries] + negatives)
            batch_scores = batch_scores.data.tolist()
            formula_predictions.extend(batch_scores)
        formula_aucs[formula] = roc_auc_score(formula_labels, np.nan_to_num(formula_predictions))
        labels.extend(formula_labels)
        predictions.extend(formula_predictions)
    overall_auc = roc_auc_score(labels, np.nan_to_num(predictions))
    return overall_auc, formula_aucs

    
def eval_perc_queries(test_queries, enc_dec, batch_size=1000, hard_negatives=False):
    perc_scores = []
    for formula in test_queries:
        formula_queries = test_queries[formula]
        offset = 0
        while offset < len(formula_queries):
            max_index = min(offset+batch_size, len(formula_queries))
            batch_queries = formula_queries[offset:max_index]
            if hard_negatives:
                lengths = [len(formula_queries[j].hard_neg_samples) for j in range(offset, max_index)]
                negatives = [n for j in range(offset, max_index) for n in formula_queries[j].hard_neg_samples]
            else:
                lengths = [len(formula_queries[j].neg_samples) for j in range(offset, max_index)]
                negatives = [n for j in range(offset, max_index) for n in formula_queries[j].neg_samples]
            offset += batch_size

            batch_scores = enc_dec.forward(formula, 
                    batch_queries+[b for i, b in enumerate(batch_queries) for _ in range(lengths[i])], 
                    [q.target_node for q in batch_queries] + negatives)
            batch_scores = batch_scores.data.tolist()
            perc_scores.extend(_get_perc_scores(batch_scores, lengths))
    return np.mean(perc_scores)

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
    elif decoder == "min":
        dec = SetIntersection(out_dims, out_dims, agg_func=torch.min)
    elif decoder == "min-simple":
        dec = SimpleSetIntersection(agg_func=torch.min)
    else:
        raise Exception("Intersection decoder not recognized.")
    return dec

def setup_logging(log_file, console=True):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
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

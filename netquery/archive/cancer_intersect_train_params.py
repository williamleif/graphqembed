import csv
import time
import random
import pickle
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

def evaluate_metapath_auc(test_metapaths, neg_metapaths, graph, enc_dec, batch_size=512):
    predictions = []
    labels = []
    for rels in test_metapaths:
        #if rels[0] == rels[1]:
        #    continue
        pos_metapaths = test_metapaths[rels]
        rels_pos_metapaths = [(e[0],e[1]) for e in pos_metapaths]
        if len(rels_pos_metapaths)>0:
            rels_neg_metapaths = [(e[0], random.choice(neg_metapaths[rels][e[0]])) for e in rels_pos_metapaths]
            labels.extend([1 for _ in rels_pos_metapaths] + [0 for _ in rels_neg_metapaths])
            metapaths = rels_pos_metapaths + rels_neg_metapaths
            splits = len(metapaths) / batch_size + 1
            for metapath_split in np.array_split(metapaths, splits):
                scores = enc_dec.forward([e[0] for e in metapath_split],
                        [e[1] for e in metapath_split], rels)
                predictions.extend(scores.data.tolist())
	    #print len(predictions), sum(np.isnan(predictions))
    try:
        auc_score =  roc_auc_score(labels, predictions)
	return auc_score
    except:
        return -1

def evaluate_metapath_margin(test_metapaths, neg_metapaths, graph, enc_dec, negative=100, batch_size=512):
    test_metapaths = [e for sub_list in test_metapaths.values() for e in sub_list]
    np.random.seed(0)
    loss = 0.
    for i, test_metapath in enumerate(test_metapaths): 
        loss += enc_dec.margin_loss([test_metapath[0] for _ in range(negative)],
                [test_metapath[1] for _ in range(negative)], test_metapath[2], "path", [random.choice(neg_metapaths[test_metapath[2]][test_metapath[0]]) for _ in range(negative)]).data[0]
    loss /= len(test_metapaths)
    return loss

def evaluate_intersect_auc(test_intersects, cancer_neg_ints, graph, enc_dec, run_all = False, batch_size=512):
    predictions = []
    labels = []
    if run_all:
        sample_rels = test_intersects.keys()
    else:
        sample_rels = np.random.choice(test_intersects.keys(), 100)
    for rels in sample_rels:    
        rels_pos_ints = test_intersects[rels]
        if len(rels_pos_ints)>0:
            if len(rels) == 3:
                #samples = [(entry[0], entry[1], entry[2], entry[3]) for entry in rels_pos_ints]
                rels_neg_ints = []
                for sample in rels_pos_ints:
                    rels_neg_ints.append((sample[0], sample[1], sample[2], random.choice(cancer_neg_ints[rels][(sample[0],sample[1], sample[2])])))
                labels.extend([1 for _ in rels_pos_ints] + [0 for _ in rels_neg_ints])
                ints = rels_pos_ints + rels_neg_ints
                splits = len(ints) / batch_size + 1
                for int_split in np.array_split(ints, splits):
                    scores = enc_dec.forward([e[0] for e in int_split],
                        [e[1] for e in int_split], rels, "intersect", [e[3] for e in int_split], [e[2] for e in int_split])
                    predictions.extend(scores.data.tolist())
                #print len(predictions), sum(np.isnan(predictions))
            else:
                #samples = [(entry[0], entry[1], entry[2]) for entry in rels_pos_ints]
                rels_neg_ints = []
                for sample in rels_pos_ints:
                    rels_neg_ints.append((sample[0], sample[1], random.choice(cancer_neg_ints[rels][(sample[0],sample[1])])))
                labels.extend([1 for _ in rels_pos_ints] + [0 for _ in rels_neg_ints])
                ints = rels_pos_ints + rels_neg_ints
                splits = len(ints) / batch_size + 1
                for int_split in np.array_split(ints, splits):
                    scores = enc_dec.forward([e[0] for e in int_split],
                        [e[1] for e in int_split], rels, "intersect", [e[2] for e in int_split])
                    predictions.extend(scores.data.tolist())
                #print predictions
	    	#print len(predictions), sum(np.isnan(predictions))
    try:
        auc_score = roc_auc_score(labels, predictions)
        return auc_score
    except:
        return -1

def evaluate_intersect_margin(test_intersects, cancer_neg_ints, graph, enc_dec, run_all = False, negative=100, batch_size=512):
    np.random.seed(0)
    loss = 0.
    len_test_intersects = 0
    if run_all:
        sample_rels = test_intersects.keys()
    else:
        sample_rels = np.random.choice(test_intersects.keys(), 100)
    for rel in sample_rels:
	rel_pos_ints = test_intersects[rel]
	if len(rel_pos_ints)>0:
	    if len(rel) == 3:
	    	for sample in rel_pos_ints:
	            loss += enc_dec.margin_loss([sample[0] for _ in range(negative)],
                	[sample[1] for _ in range(negative)], rel, "intersect", [random.choice(cancer_neg_ints[rel][(sample[0],sample[1],sample[2])]) for _ in range(negative)],
			[sample[3] for _ in range(negative)], [sample[2] for _ in range(negative)]).data[0]
	    	    len_test_intersects += 1
	    else:
		for sample in rel_pos_ints:
                    loss += enc_dec.margin_loss([sample[0] for _ in range(negative)],
                        [sample[1] for _ in range(negative)], rel, "intersect", [random.choice(cancer_neg_ints[rel][(sample[0],sample[1])]) for _ in range(negative)],
                        [sample[2] for _ in range(negative)]).data[0]
		    len_test_intersects += 1
    loss /= len_test_intersects
    return loss

    #Parallel implementation
    '''
    np.random.seed(0)
    loss = 0.
    len_test_intersects = 0
    if run_all:
	sample_rels = test_intersects.keys()
    else:
    	sample_rels = np.random.choice(test_intersects.keys(), 100)
    for rel in sample_rels:
	if len(test_intersects[rel]) > 0:
            if len(rel) == 3:
		rel_pos_ints = test_intersects[rel]
                samples = [(entry[0], entry[1], entry[2], entry[3], random.choice(cancer_neg_ints[rel][(entry[0],entry[1],entry[2])])) for entry in rel_pos_ints]
                splits = len(samples) / batch_size + 1
                for sample_split in np.array_split(samples, splits):
                    loss += enc_dec.margin_loss([e[0] for e in sample_split  for _ in range(negative)],
                        [e[1] for e in sample_split for _ in range(negative)], rel, "intersect", [e[4] for e in sample_split  for _ in range(negative)],
                        [e[3] for e in sample_split for _ in range(negative)], [e[2] for e in sample_split for _ in range(negative)]).data[0]*len(sample_split)
		    len_test_intersects += len(sample_split)
	    else:
	    	rel_pos_ints = test_intersects[rel]
            	samples = [(entry[0], entry[1], entry[2], random.choice(cancer_neg_ints[rel][(entry[0],entry[1])])) for entry in rel_pos_ints]
            	splits = len(samples) / batch_size + 1
            	for sample_split in np.array_split(samples, splits):
                    loss += enc_dec.margin_loss([e[0] for e in sample_split  for _ in range(negative)],
                    	[e[1] for e in sample_split for _ in range(negative)], rel, "intersect", [e[3] for e in sample_split  for _ in range(negative)],
                    	[e[2] for e in sample_split for _ in range(negative)]).data[0]*len(sample_split)
		    len_test_intersects += len(sample_split)
    print loss, len_test_intersects
    loss /= len_test_intersects  
    return loss
    '''

def evaluate_edge_auc(test_edges, neg_edges, graph, enc_dec, batch_size=512):
    predictions = []
    labels = []
    for rel in test_edges:
        rel_pos_edges = test_edges[rel]
        rel_neg_edges = [(e[0], random.choice(neg_edges[(rel,)][e[0]])) for e in rel_pos_edges]
        labels.extend([1 for _ in rel_pos_edges] + [0 for _ in rel_neg_edges])
        edges = rel_pos_edges + rel_neg_edges
        splits = len(edges) / batch_size + 1
        for edge_split in np.array_split(edges, splits):
            scores = enc_dec.forward([e[0] for e in edge_split],
                    [e[1] for e in edge_split], [rel])
            predictions.extend(scores.data.tolist())
    try:
        auc_score =  roc_auc_score(labels, predictions)
	return auc_score
    except:
        return -1

def evaluate_edge_margin(test_edges, neg_edges, graph, enc_dec, negative=100, batch_size=512):
    test_edges = [e for sub_list in test_edges.values() for e in sub_list]
    np.random.seed(0)
    loss = 0.
    for i, test_edge in enumerate(test_edges): 
        loss += enc_dec.margin_loss([test_edge[0] for _ in range(negative)],
                [test_edge[1] for _ in range(negative)], [test_edge[2]], "path", [random.choice(neg_edges[(test_edge[2],)][test_edge[0]]) for _ in range(negative)]).data[0]
    loss /= len(test_edges)
    return loss


def get_decoder(graph, out_dims, decoder):
    if decoder == "bilinear":
        dec = DotBilinearMetapathDecoder(graph.relations, out_dims)
    elif decoder == "transe":
        dec = TransEMetapathDecoder(graph.relations, out_dims)
    elif decoder == "bilinear-diag":
        dec = BilinearDiagMetapathDecoder(graph.relations, out_dims)
    return dec

def train(feature_dim, lr_edge, lr_metapath, lr_int, model, batch_size, max_batches, tol, max_path_len, cuda, results, decoder, opt, agg):
    feature_dim = 16
    relations, adj_lists, node_maps = pickle.load(open("/dfs/scratch0/netquery/cancer.pkl"))
    # relations, adj_lists, node_maps = pickle.load(open("cancer.pkl"))
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
        feature_module.weight.data.normal_(0, 1./feature_dim)

    if cuda:
        features = lambda nodes, mode: feature_modules[mode].forward(
                Variable(torch.LongTensor([node_maps[mode][node] for node in nodes])).cuda())
    else:
        features = lambda nodes, mode: feature_modules[mode].forward(
                Variable(torch.LongTensor([node_maps[mode][node] for node in nodes])))

    graph = Graph(features, feature_dims, relations, adj_lists)

    #Create chains and intersections on entire graph
    cancer_chains, cancer_neg_chains = graph.create_chains_byrels()
    cancer_pos_ints, cancer_neg_ints = graph.create_intersections_byrels()

    #Create all edges, metapaths and intersections
    edges = {}
    metapaths = {}
    for rel in cancer_chains:
        if len(rel) == 1:
            edges[rel[0]] = [(node1, node2, rel[0]) for node1 in cancer_chains[rel] for node2 in cancer_chains[rel][node1]]
        elif len(rel) in [2,3]:
            metapaths[rel] = [(node1, entry[-1], rel) for node1 in cancer_chains[rel] for entry in cancer_chains[rel][node1]]
    
    for edge_list in edges.values():
            random.shuffle(edge_list)
    for metapath_list in metapaths.values():
            random.shuffle(metapath_list)
    
    pos_ints = {}
    for rel in cancer_pos_ints:
        if len(rel) == 3:
            pos_ints[rel] = [(node1, node2, node3, target) for (node1, node2, node3) in cancer_pos_ints[rel] for target in cancer_pos_ints[rel][(node1,node2,node3)]]
        else:
            pos_ints[rel] = [(node1, node2, target) for (node1, node2) in cancer_pos_ints[rel] for target in cancer_pos_ints[rel][(node1,node2)]]
    #Get test edges and remove them from the graph
    train_edges = {rel:edge_list[:int(0.9*len(edge_list))] for rel, edge_list in edges.iteritems()}
    test_edges = {rel:edge_list[int(0.9*len(edge_list)):] for rel, edge_list in edges.iteritems()}
    graph.remove_edges([e for edge_list in test_edges.values() for e in edge_list])

    #Create TRAIN chains and metapaths from the train graph (test edges removed)
    train_cancer_chains, train_cancer_neg_chains = graph.create_chains_byrels()
    train_cancer_pos_ints, train_cancer_neg_ints = graph.create_intersections_byrels()

    #Create TRAINING metapaths and intersections from the train graph
    train_metapaths={}
    for rel in train_cancer_chains:
	if len(rel) in [2,3]:
            train_metapaths[rel] = [(node1, entry[-1], rel) for node1 in train_cancer_chains[rel] for entry in train_cancer_chains[rel][node1]]
    train_pos_ints = {}
    for rel in train_cancer_pos_ints:
        if len(rel) == 3:
            train_pos_ints[rel] = [(node1, node2, node3, target) for (node1, node2, node3) in train_cancer_pos_ints[rel] for target in train_cancer_pos_ints[rel][(node1,node2,node3)]]
        else:
            train_pos_ints[rel] = [(node1, node2, target) for (node1, node2) in train_cancer_pos_ints[rel] for target in train_cancer_pos_ints[rel][(node1,node2)]]

    #Create test metapaths and test intersections be removinf training metapaths and intersections from the full sets respectively
    test_metapaths = {rel:list(set(metapaths[rel]) - set(train_metapaths[rel])) for rel in metapaths}
    test_ints = {rel:list(set(pos_ints[rel]) - set(train_pos_ints[rel])) for rel in pos_ints}
    
    '''
    with open("train_edges.pkl","wb") as f:
    	pickle.dump(train_edges, f)
    with open("test_edges.pkl", "wb") as f:
    	pickle.dump(test_edges, f)
    with open("train_metapaths.pkl", "wb") as f:
    	pickle.dump(train_metapaths, f)
    with open("test_metapaths.pkl", "wb") as f:
    	pickle.dump(test_metapaths, f)
    with open("train_ints.pkl", "wb") as f:
   	pickle.dump(train_pos_ints, f)
    with open("test_ints.pkl", "wb") as f:
    	pickle.dump(test_ints, f)
    '''

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
    
    inter_dec = MinIntersection(feature_dims.keys(), feature_dims, feature_dims)
    combined_enc_dec = LogCombinedEncoderDecoder(graph, enc, dec, inter_dec)
    if cuda:
        combined_enc_dec.cuda()
    
    
    print "Checking eval functions"
    beg_int_auc =  evaluate_intersect_auc(test_ints, cancer_neg_ints, graph, combined_enc_dec, True)
    #beg_int_loss = evaluate_intersect_margin(test_ints, cancer_neg_ints, graph, combined_enc_dec, False)
    beg_path_auc = evaluate_metapath_auc(test_metapaths, cancer_neg_chains, graph, combined_enc_dec, batch_size=batch_size)
    beg_edge_auc = evaluate_edge_auc(test_edges, cancer_neg_chains, graph, combined_enc_dec)
    #beg_edge_loss = evaluate_edge_margin(test_edges, cancer_neg_chains, graph, combined_enc_dec)
    #beg_path_loss = evaluate_metapath_margin(test_metapaths, cancer_neg_chains, graph, combined_enc_dec)
    print beg_edge_auc, beg_path_auc#, beg_int_auc, beg_edge_loss, beg_path_loss, beg_int_loss
    

    losses = []
    ema_loss = None
    
    if opt == "sgd":
        optimizer = optim.SGD(combined_enc_dec.parameters(), lr=lr_edge, momentum=0.000)
    elif opt == "sgd-momentum":
        optimizer = optim.SGD(combined_enc_dec.parameters(), lr=lr_edge, momentum=0.9)
    elif opt == "adam":
        optimizer = optim.Adam(combined_enc_dec.parameters(), lr=lr_edge)

    conv = -1
    for i in range(max_batches):
        rel = graph.sample_relation()
        #print len(train_edges[rel])
        start = random.randint(0, max(0,len(train_edges[rel])-batch_size))
        edges = train_edges[rel][start:start+batch_size]
        if len(edges) == 0:
            continue
        optimizer.zero_grad()
        #combined_enc_dec.graph.remove_edges(edges)
        neg_nodes = [random.choice(train_cancer_neg_chains[(rel,)][e[0]]) for e in edges]
        loss = combined_enc_dec.margin_loss([edge[0] for edge in edges], [edge[1] for edge in edges], [rel], "path", neg_nodes)
        #combined_enc_dec.graph.add_edges(edges)
        losses.append(loss.data[0])
        if ema_loss == None:
            ema_loss = loss.data[0]
        else:
            ema_loss = 0.99*ema_loss + 0.01*loss.data[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm(combined_enc_dec.parameters(), 0.00001)
        optimizer.step()
        if i % 100 == 0:
            print i, ema_loss
        if i > 2000 and i % 100 == 0:
            conv = np.mean(losses[i-2000:i-1000]) - np.mean(losses[i-1000:i]) 
            print "conv", conv
            if conv < tol:
                break
    print "After training on edges:"
    print combined_enc_dec.edge_dec.mats
    train1_edge_auc = evaluate_edge_auc(test_edges, cancer_neg_chains, graph, combined_enc_dec)
    #train1_edge_loss = evaluate_edge_margin(test_edges, cancer_neg_chains, graph, combined_enc_dec)
    #train1_path_loss = evaluate_metapath_margin(test_metapaths, cancer_neg_chains, graph, combined_enc_dec)
#    train1_path_auc = evaluate_metapath_auc(test_metapaths, cancer_neg_chains, graph, combined_enc_dec, batch_size=batch_size)
    train1_int_auc =  evaluate_intersect_auc(test_ints, cancer_neg_ints, graph, combined_enc_dec, True)
    #train1_int_loss = evaluate_intersect_margin(test_ints, cancer_neg_ints, graph, combined_enc_dec, False)
    print train1_edge_auc, train1_int_auc#, train1_edge_loss, train1_path_loss, train1_int_loss
    
    """ 
    losses = []
    ema_loss = None
    if opt == "sgd":
        optimizer = optim.SGD(combined_enc_dec.parameters(), lr=lr_metapath, momentum=0.000)
    elif opt == "sgd-momentum":
        optimizer = optim.SGD(combined_enc_dec.parameters(), lr=lr_metapath, momentum=0.9)
    elif opt == "adam":
        optimizer = optim.Adam(combined_enc_dec.parameters(), lr=lr_metapath)

    conv = -1
    for i in range(max_batches):
        while True:
            rels = graph.sample_metapath()
            if len(train_metapaths[rels]) > 0:
                break
        start = random.randint(0, max(0,len(train_metapaths[rels])-batch_size))
        edges = train_metapaths[rels][start:start+batch_size]
        nodes1 = [edge[0] for edge in edges]
        nodes2 = [edge[1] for edge in edges]
        neg_nodes = [random.choice(train_cancer_neg_chains[rels][e[0]]) for e in edges]
        optimizer.zero_grad()
        loss = combined_enc_dec.margin_loss(nodes1, nodes2, rels, "path", neg_nodes)
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
            print "MRR:", evaluate_edge_auc(test_edges, cancer_neg_chains, graph, combined_enc_dec)

    print "After training on metapaths:"
    train2_edge_auc = evaluate_edge_auc(test_edges, cancer_neg_chains, graph, combined_enc_dec)
    #train2_edge_loss = evaluate_edge_margin(test_edges, cancer_neg_chains, graph, combined_enc_dec)
    #train2_path_loss = evaluate_metapath_margin(test_metapaths, cancer_neg_chains, graph, combined_enc_dec)
    train2_path_auc = evaluate_metapath_auc(test_metapaths, cancer_neg_chains, graph, combined_enc_dec, batch_size=batch_size)
    #train2_int_auc =  evaluate_intersect_auc(test_ints, cancer_neg_ints, graph, combined_enc_dec, False)
    #train2_int_loss = evaluate_intersect_margin(test_ints, cancer_neg_ints, graph, combined_enc_dec, False)
    print train2_edge_auc, train2_path_auc#, train2_int_auc, train2_edge_loss, train2_path_loss, train2_int_loss
    """
    
    losses = []
    ema_loss = None
    if opt == "sgd":
        optimizer = optim.SGD(combined_enc_dec.inter_dec.parameters(), lr=lr_int, momentum=0.000)
    elif opt == "sgd-momentum":
        optimizer = optim.SGD(combined_enc_dec.parameters(), lr=lr_int, momentum=0.9)
    elif opt == "adam":
        optimizer = optim.Adam(combined_enc_dec.parameters(), lr=lr_int)

    conv = -1
    for i in range(max_batches):
        while True:
            rels = graph.sample_intersection()
            random.shuffle(train_pos_ints[rels])
            samples = train_pos_ints[rels][:batch_size]
            if len(samples) > 0:
                break
        query_nodes1 = [edge[0] for edge in samples]
        query_nodes2 = [edge[1] for edge in samples]
        if len(rels)==3:
            query_nodes3 = [edge[2] for edge in samples]
            target_nodes = [edge[3] for edge in samples]
            neg_nodes = [random.choice(train_cancer_neg_ints[rels][(query_nodes1[j],query_nodes2[j],query_nodes3[j])]) for j in range(len(samples))]
        else:
            query_nodes3 = []
            target_nodes = [edge[2] for edge in samples]
            neg_nodes = [random.choice(train_cancer_neg_ints[rels][(query_nodes1[j],query_nodes2[j])]) for j in range(len(samples))]
        
        optimizer.zero_grad()
        loss = combined_enc_dec.margin_loss(query_nodes1, query_nodes2, rels, "intersect", neg_nodes, target_nodes, query_nodes3)
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
            print "MRR:", evaluate_edge_auc(test_edges, cancer_neg_chains, graph, combined_enc_dec)
            print "Inersection AUC:", evaluate_intersect_auc(test_ints, cancer_neg_ints, graph, combined_enc_dec, True)
    
    print "After training on intersections:"    
    train3_edge_auc = evaluate_edge_auc(test_edges, cancer_neg_chains, graph, combined_enc_dec)
#    train3_edge_loss = evaluate_edge_margin(test_edges, cancer_neg_chains, graph, combined_enc_dec)
    #train3_path_loss = evaluate_metapath_margin(test_metapaths, cancer_neg_chains, graph, combined_enc_dec)
    #train3_path_auc = evaluate_metapath_auc(test_metapaths, cancer_neg_chains, graph, combined_enc_dec, batch_size=batch_size)
    train3_int_auc =  evaluate_intersect_auc(test_ints, cancer_neg_ints, graph, combined_enc_dec, True)
    train3_int_auc_train =  evaluate_intersect_auc(train_pos_ints, train_cancer_neg_ints, graph, combined_enc_dec, True)
#    train3_int_loss = evaluate_intersect_margin(test_ints, cancer_neg_ints, graph, combined_enc_dec, True)
    print train3_edge_auc, train3_int_auc, train3_int_auc_train#, train3_edge_loss, train3_path_loss, train3_int_loss

    with open (results, "a") as csvfile:
        writer = csv.writer(csvfile)
	writer.writerow([str(lr_edge), str(lr_metapath), str(beg_edge_auc), str(beg_path_auc), str(train1_edge_auc), str(train1_path_auc), str(train2_edge_auc), str(train2_path_auc)]) 
        #writer.writerow([str(lr), str(beg_edge_auc), str(beg_path_auc), str(beg_int_auc), str(beg_edge_loss), str(beg_path_loss), str(beg_int_loss), str(train1_edge_auc), str(train1_path_auc), str(train1_int_auc), str(train1_edge_loss), str(train1_path_loss), str(train1_int_loss),  str(train2_edge_auc), str(train2_path_auc), str(train2_int_auc), str(train2_edge_loss), str(train2_path_loss), str(train2_int_loss), str(train3_edge_auc), str(train3_path_auc), str(train3_int_auc), str(train3_edge_loss), str(train3_path_loss), str(train3_int_loss)]) 
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--lr_edge", type=float, default=0.5)
    parser.add_argument("--lr_metapath", type=float, default=0.5)
    parser.add_argument("--lr_int", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="direct")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_batches", type=int, default=100000)
    parser.add_argument("--max_path_len", type=int, default=2)
    parser.add_argument("--tol", type=float, default=0.0001)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--results", type=str, default="results.csv")
    parser.add_argument("--decoder", type=str, default="bilinear")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--aggregator", type=str, default="mean")
    args = parser.parse_args()
    train(args.feature_dim, args.lr_edge, args.lr_metapath, args.lr_int, args.model, args.batch_size, args.max_batches, args.tol, args.max_path_len, args.cuda, args.results, args.decoder, args.optimizer, args.aggregator)

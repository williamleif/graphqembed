import cPickle as pickle
import torch
from collections import OrderedDict, defaultdict
from multiprocessing import Process
import random
import json
from netquery.data_utils import parallel_sample, load_queries_by_type, sample_clean_test

from netquery.graph import Graph, Query, _reverse_edge

def load_graph(data_dir, embed_dim):
    rels, adj_lists, node_maps = pickle.load(open(data_dir+"/graph_data.pkl", "rb"))
    node_maps = {m : {n : i for i, n in enumerate(id_list)} for m, id_list in node_maps.iteritems()}
    for m in node_maps:
        node_maps[m][-1] = -1
    feature_dims = {m : embed_dim for m in rels}
    feature_modules = {m : torch.nn.Embedding(len(node_maps[m])+1, embed_dim) for m in rels}
    for mode in rels:
        feature_modules[mode].weight.data.normal_(0, 1./embed_dim)
    features = lambda nodes, mode : feature_modules[mode](
            torch.autograd.Variable(torch.LongTensor([node_maps[mode][n] for n in nodes])+1))
    graph = Graph(features, feature_dims, rels, adj_lists)
    return graph, feature_modules, node_maps

def sample_new_clean(data_dir):
    graph_loader = lambda : load_graph(data_dir, 10)[0]
    sample_clean_test(graph_loader, data_dir) 

def clean_test():
    test_edges = pickle.load(open("/dfs/scratch0/nqe-bio/test_edges.pkl", "rb"))
    val_edges = pickle.load(open("/dfs/scratch0/nqe-bio/val_edges.pkl", "rb"))  
    deleted_edges = set([q[0][1] for q in test_edges] + [_reverse_edge(q[0][1]) for q in test_edges] + 
                [q[0][1] for q in val_edges] + [_reverse_edge(q[0][1]) for q in val_edges])

    for i in range(2,4):
        for kind in ["val", "test"]:
            if kind == "val":
                to_keep = 1000
            else:
                to_keep = 10000
            test_queries = load_queries_by_type("/dfs/scratch0/nqe-bio/{:s}_queries_{:d}-split.pkl".format(kind, i), keep_graph=True)
            print "Loaded", i, kind
            for query_type in test_queries:
                test_queries[query_type] = [q for q in test_queries[query_type] if len(q.get_edges().intersection(deleted_edges)) > 0]
                test_queries[query_type] = test_queries[query_type][:to_keep]
            test_queries = [q.serialize() for queries in test_queries.values() for q in queries]
            pickle.dump(test_queries, open("/dfs/scratch0/nqe-bio/{:s}_queries_{:d}-clean.pkl".format(kind, i), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            print "Finished", i, kind
        


def make_train_test_edge_data(data_dir):
    print "Loading graph..."
    graph, _, _ = load_graph(data_dir, 10)
    print "Getting all edges..."
    edges = graph.get_all_edges()
    split_point = int(0.1*len(edges))
    val_test_edges = edges[:split_point]
    print "Getting negative samples..."
    val_test_edge_negsamples = [graph.get_negative_edge_samples(e, 100) for e in val_test_edges]
    print "Making and storing test queries."
    val_test_edge_queries = [Query(("1-chain", val_test_edges[i]), val_test_edge_negsamples[i], None, 100) for i in range(split_point)]
    val_split_point = int(0.1*len(val_test_edge_queries))
    val_queries = val_test_edge_queries[:val_split_point]
    test_queries = val_test_edge_queries[val_split_point:]
    pickle.dump([q.serialize() for q in val_queries], open(data_dir+"/val_edges.pkl", "w"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in test_queries], open(data_dir+"/test_edges.pkl", "w"), protocol=pickle.HIGHEST_PROTOCOL)

    print "Removing test edges..."
    graph.remove_edges(val_test_edges)
    print "Making and storing train queries."
    train_edges = graph.get_all_edges()
    train_queries = [Query(("1-chain", e), None, None) for e in train_edges]
    pickle.dump([q.serialize() for q in train_queries], open(data_dir+"/train_edges.pkl", "w"), protocol=pickle.HIGHEST_PROTOCOL)

def _discard_negatives(file_name, small_prop=0.9):
    queries = pickle.load(open(file_name, "rb"))
#    queries = [q if random.random() > small_prop else (q[0],[random.choice(tuple(q[1]))], None if q[2] is None else [random.choice(tuple(q[2]))]) for q in queries]
    queries = [q if random.random() > small_prop else (q[0],[random.choice(list(q[1]))], None if q[2] is None else [random.choice(list(q[2]))]) for q in queries] 
    pickle.dump(queries, open(file_name.split(".")[0] + "-split.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print "Finished", file_name

def discard_negatives(data_dir):
    _discard_negatives(data_dir + "/val_edges.pkl")
    _discard_negatives(data_dir + "/test_edges.pkl")
    for i in range(2,4):
        _discard_negatives(data_dir + "/val_queries_{:d}.pkl".format(i))
        _discard_negatives(data_dir + "/test_queries_{:d}.pkl".format(i))


def make_train_test_query_data(data_dir):
    graph, _, _ = load_graph(data_dir, 10)
    queries_2, queries_3 = parallel_sample(graph, 20, 50000, data_dir, test=False)
    t_queries_2, t_queries_3 = parallel_sample(graph, 20, 5000, data_dir, test=True)
    t_queries_2 = list(set(t_queries_2) - set(queries_2))
    t_queries_3 = list(set(t_queries_3) - set(queries_3))
    pickle.dump([q.serialize() for q in queries_2], open(data_dir + "/train_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in queries_3], open(data_dir + "/train_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in t_queries_2[10000:]], open(data_dir + "/test_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in t_queries_3[10000:]], open(data_dir + "/test_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in t_queries_2[:10000]], open(data_dir + "/val_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in t_queries_3[:10000]], open(data_dir + "/val_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    #make_train_test_query_data("/dfs/scratch0/nqe-bio/")
    #make_train_test_edge_data("/dfs/scratch0/nqe-bio/")
    sample_new_clean("/dfs/scratch0/nqe-bio/")
    #clean_test()

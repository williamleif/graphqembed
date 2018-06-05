from collections import defaultdict
import cPickle as pickle
from multiprocessing import Process
from netquery.graph import Query

def load_queries(data_file, keep_graph=False):
    raw_info = pickle.load(open(data_file, "rb"))
    return [Query.deserialize(info, keep_graph=keep_graph) for info in raw_info]

def load_queries_by_formula(data_file):
    raw_info = pickle.load(open(data_file, "rb"))
    queries = defaultdict(lambda : defaultdict(list))
    for raw_query in raw_info:
        query = Query.deserialize(raw_query)
        queries[query.formula.query_type][query.formula].append(query)
    return queries

def load_queries_by_type(data_file, keep_graph=True):
    raw_info = pickle.load(open(data_file, "rb"))
    queries = defaultdict(list)
    for raw_query in raw_info:
        query = Query.deserialize(raw_query, keep_graph=keep_graph)
        queries[query.formula.query_type].append(query)
    return queries


def load_test_queries_by_formula(data_file):
    raw_info = pickle.load(open(data_file, "rb"))
    queries = {"full_neg" : defaultdict(lambda : defaultdict(list)), 
            "one_neg" : defaultdict(lambda : defaultdict(list))}
    for raw_query in raw_info:
        neg_type = "full_neg" if len(raw_query[1]) > 1 else "one_neg"
        query = Query.deserialize(raw_query)
        queries[neg_type][query.formula.query_type][query.formula].append(query)
    return queries

def sample_clean_test(graph_loader, data_dir):
    train_graph = graph_loader()
    test_graph = graph_loader()
    test_edges = load_queries(data_dir + "/test_edges.pkl")
    val_edges = load_queries(data_dir + "/val_edges.pkl")
    train_graph.remove_edges([(q.target_node, q.formula.rels[0], q.anchor_nodes[0]) for q in test_edges+val_edges])
    test_queries_2 = test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], 9000, 1)
    test_queries_2.extend(test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], 1000, 1000))
    val_queries_2 = test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], 10, 900)
    val_queries_2.extend(test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], 100, 1000))
    val_queries_2 = list(set(val_queries_2)-set(test_queries_2))
    print len(val_queries_2)
    test_queries_3 = test_graph.sample_test_queries(train_graph, ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"], 9000, 1)
    test_queries_3.extend(test_graph.sample_test_queries(train_graph, ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"], 1000, 1000))
    val_queries_3 = test_graph.sample_test_queries(train_graph, ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"], 900, 1)
    val_queries_3.extend(test_graph.sample_test_queries(train_graph, ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"], 100, 1000))
    val_queries_3 = list(set(val_queries_3)-set(test_queries_3))
    print len(val_queries_3)
    pickle.dump([q.serialize() for q in test_queries_2], open(data_dir + "/test_queries_2-newclean.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in test_queries_3], open(data_dir + "/test_queries_3-newclean.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in val_queries_2], open(data_dir + "/val_queries_2-newclean.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in val_queries_3], open(data_dir + "/val_queries_3-newclean.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        
def clean_test(train_queries, test_queries):
    for query_type in train_queries:
        train_set = set(train_queries[query_type])
        test_queries[query_type] = [q for q in test_queries[query_type] if not q in train_set]
    return test_queries

def parallel_sample_worker(pid, num_samples, graph, data_dir, is_test, test_edges):
    if not is_test:
        graph.remove_edges([(q.target_node, q.formula.rels[0], q.anchor_nodes[0]) for q in test_edges])
    print "Running worker", pid
    queries_2 = graph.sample_queries(2, num_samples, 100 if is_test else 1, verbose=True)
    queries_3 = graph.sample_queries(3, num_samples, 100 if is_test else 1, verbose=True)
    print "Done running worker, now saving data", pid
    pickle.dump([q.serialize() for q in queries_2], open(data_dir + "/queries_2-{:d}.pkl".format(pid), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in queries_3], open(data_dir + "/queries_3-{:d}.pkl".format(pid), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

def parallel_sample(graph, num_workers, samples_per_worker, data_dir, test=False, start_ind=None):
    if test:
        print "Loading test/val data.."
        test_edges = load_queries(data_dir + "/test_edges.pkl")
        val_edges = load_queries(data_dir + "/val_edges.pkl")
        proc_range = range(num_workers) if start_ind is None else range(start_ind, num_workers+start_ind)
        procs = [Process(target=parallel_sample_worker, args=[i, samples_per_worker, graph, data_dir, test, val_edges+test_edges]) for i in proc_range]
        for p in procs:
            p.start()
        for p in procs:
            p.join() 
    queries_2 = []
    queries_3 = []
    for i in range(num_workers):
        new_queries_2 = load_queries(data_dir+"/queries_2-{:d}.pkl".format(i), keep_graph=True)
        queries_2.extend(new_queries_2)
        new_queries_3 = load_queries(data_dir+"/queries_3-{:d}.pkl".format(i), keep_graph=True)
        queries_3.extend(new_queries_3)
    return queries_2, queries_3

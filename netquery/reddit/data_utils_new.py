import numpy as np
from collections import defaultdict, Counter, OrderedDict
import cPickle as pickle
from multiprocessing import Process
import os

from gensim.models.word2vec import Word2Vec

import spacy
import random

import torch
import torch.nn as nn
from netquery.graph import Graph, Query, _reverse_edge
from netquery.data_utils import parallel_sample, load_queries_by_type, sample_clean_test

def load_comments(filename, valid_days,  post_ids, user_ids):
    adj_lists = {("user", "comment", "post") : defaultdict(set),
                 ("post", "comment", "user") : defaultdict(set)}

    with open(filename) as fp:
        for i, line in enumerate(fp):
            if i % 1000 == 0:
                print "Done comment", i
 
            info = line.strip().split(",")
            if not int(info[0].split("-")[-1].split()[0]) in valid_days:
                continue
            post = info[-1]
            if not post in post_ids:
                continue
            user = info[1]
            if not user in user_ids:
                user_ids[user] = len(user_ids)
            user_id = user_ids[user]
            adj_lists[("user", "comment", "post")][user_id].add(post_ids[post])
            adj_lists[("post", "comment", "user")][post_ids[post]].add(user_id)
    return adj_lists

def clean_words(post_words, word_counts, min_count=1):
    word_map = {}
    def _get_id(word):
        if word in word_map:
            return word_map[word]
        elif word_counts[word] >= min_count:
            word_map[word] = len(word_map)
            return word_map[word]
        else:
            return None
    for post in post_words:
        post_words[post] = set([_get_id(w) for w in post_words[post] if _get_id(w) != None])
    print "Kept", len(word_map), "words"
    return post_words

def load_posts(filename, valid_days, w2v, sub_limit=None, subs_set=None):

    post_ids = {}
    sub_belongs_to = {}
    made_by = {}
    post_feats = {}
    word_dict = Counter()

    subs = Counter()
    with open(filename) as fp:
        for i, line in enumerate(fp):
            info = line.split(",")
            subs[info[2]] += 1
    if not subs_set is None: 
        subs = set([sub for sub in subs if sub.lower() in subs_set])
        print "Kept", len(subs), "subs"
    elif not sub_limit is None:
        subs = set(random.sample([sub for sub, count in subs.iteritems() if count >= 10], sub_limit))
        print "Kept", len(subs), "subs"

    with open(filename) as fp:
        for i, line in enumerate(fp):
            if i % 1000 == 0:
                print "Done post", i

            info = line.split(",")
            if not int(info[0].split("-")[-1].split()[0]) in valid_days:
                continue
            if (not sub_limit is None or not subs_set is None) and not info[2] in subs:
                continue
            post_id = len(post_ids)
            post_ids[info[3]] = post_id
            sub_belongs_to[post_id] = info[2]
            made_by[post_id] = info[1]
            text = ". ".join(info[-3:])
            tokenized = tokenizer(unicode(text, errors="ignore"))
            words = set([w.lower for w in tokenized if w.lower_])
            post_feats[post_id] = words
            word_dict.update(words)
    post_feats = clean_words(post_feats, word_dict, min_count=1)
    return post_ids, sub_belongs_to, made_by, post_feats

def load_votes(filename, valid_days, posts, user_ids):
    adj_lists = {
                 ("user", "up", "post") : defaultdict(set),
                 ("user", "down", "post") : defaultdict(set),
                 ("post", "up", "user") : defaultdict(set),
                 ("post", "down", "user") : defaultdict(set),
                 }

    with open(filename) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split(",")
            if not int(info[0].split("-")[-1].split()[0]) in valid_days or info[-1] == "clear":
                continue
            vote_type = "comment" if info[-2] == "comment" else "post" 
            if vote_type == "post" and info[-3] in posts:
                if not info[1] in user_ids:
                    user_ids[info[1]] = len(user_ids)
                user_id = user_ids[info[1]]
                post_id = posts[info[-3]]
                adj_lists[("user", info[-1], "post")][user_id].add(post_id)
                adj_lists[("post", info[-1], "user")][post_id].add(user_id)

    return adj_lists

def load_subscriptions(user_ids, sub_ids):
    adj_lists = {
                 ("user", "subscribe", "community") : defaultdict(set),
                 ("community", "subscribe", "user") : defaultdict(set),
                 }

    for f in os.listdir("/dfs/dataset/infolab/20180122-Reddit/data/stanford_subscription_data/"):
        with open("/dfs/dataset/infolab/20180122-Reddit/data/stanford_subscription_data/"+f, "r") as fp:
            fp.readline()
            for line in fp:
                info = line.strip().split(",")
                date = info[0].split()[0].split("-")
                if int(date[1]) > 5:
                    continue
                if info[-1] == "subscribe" and info[1] in user_ids and info[2] in sub_ids:
                    user = user_ids[info[1]]
                    sub = sub_ids[info[2]]
                    adj_lists[("user", "subscribe", "community")][user].add(sub)
                    adj_lists[("community", "subscribe", "user")][sub].add(user)
    return adj_lists


def load_graph(info_dir, embed_dim=16, cuda=False):
    print "Loading adjacency info..."
    adj_lists = pickle.load(open(info_dir + "/adj_lists.pkl"))
    relations = pickle.load(open(info_dir + "/rels.pkl"))
    post_words = pickle.load(open(info_dir + "/post_words.pkl"))

    num_users = len(set([id for rel, adj in adj_lists.iteritems() for id in adj if rel[0] == "user"]))
    num_communities = len(set([id for rel, adj in adj_lists.iteritems() for id in adj if rel[0] == "community"]))
    num_words = len(set([w for words in post_words.values() for w in words]))
    post_words = {post : torch.LongTensor([w for w in post_words[post]]) for post in post_words}

    feature_modules = {
        "post" : nn.EmbeddingBag(num_words, embed_dim), 
        "user" : nn.Embedding(num_users+1, embed_dim), 
        "community" : nn.Embedding(num_communities+1, embed_dim), 
    }
    for mode in feature_modules:
        feature_modules[mode].weight.data.normal_(0, 1./embed_dim)
    if not cuda:
        def _feature_func(nodes, mode):
            if mode != "post":
                return feature_modules[mode](
                    torch.autograd.Variable(torch.LongTensor(nodes)+1))
            else:
                offsets = np.concatenate(([0], np.cumsum([post_words[post].size()[0] for post in nodes[:-1]])))
                return feature_modules[mode](torch.autograd.Variable(torch.cat([post_words[post] for post in nodes])),
                        torch.autograd.Variable(torch.LongTensor(offsets)))
    else:
        def _feature_func(nodes, mode):
            if mode != "post":
                return feature_modules[mode](
                    torch.autograd.Variable(torch.LongTensor(nodes)+1).cuda())
            else:
                offsets = np.concatenate(([0], np.cumsum([post_words[post].size()[0] for post in nodes[:-1]])))
                return feature_modules[mode](torch.autograd.Variable(torch.cat([post_words[post] for post in nodes])).cuda(),
                        torch.autograd.Variable(torch.LongTensor(offsets)).cuda())

    feature_dims = {mode : embed.weight.size()[1] for mode, embed in feature_modules.iteritems()}
    graph = Graph(_feature_func, feature_dims, relations, adj_lists)
    return graph, feature_modules

def build_graph(fn, include_days, sub_limit=None, subs_set=None):
    nlp = spacy.load('en')
    global tokenizer 
    tokenizer = nlp.tokenizer

    w2v = Word2Vec.load("/dfs/scratch0/reddit/w2v_model_021518_2016_full.gsim")
    #id, sub, author, words
    test_post_info = load_posts("/dfs/scratch0/nqe-reddit-new/post_data/"+fn, include_days, w2v, sub_limit=sub_limit, subs_set=subs_set)
    
    relations = {
            "user" : [("post", "up"), ("post", "down"), ("post", "make"), ("post", "comment"), ("community", "subscribe")],
            "post" : [("user", "up"), ("user", "down"), ("user", "make"), ("user", "comment"), ("community", "belong")],
            "community" : [("post", "belong"), ("user", "subscribe")],
            }
    adj_lists = {}
    for mode, rels in relations.iteritems():
        for rel in rels:
            adj_lists[(mode, rel[-1], rel[0])] = defaultdict(set)

    sub_ids = {}
    user_ids = {}
    for post, sub in test_post_info[1].iteritems():
        if not sub in sub_ids:
            sub_ids[sub] = len(sub_ids)
        sub_id = sub_ids[sub]
        adj_lists[("post", "belong", "community")][post].add(sub_id)
        adj_lists[("community", "belong", "post")][sub_id].add(post)
    for post, user in test_post_info[2].iteritems():
        if not user in user_ids:
            user_ids[user] = len(user_ids)
        user_id = user_ids[user]
        adj_lists[("user", "make", "post")][user_id].add(post)
        adj_lists[("post", "make", "user")][post].add(user_id)

    print "Loading comments.."
    comment_adjs = load_comments("/dfs/scratch0/nqe-reddit-new/comment_data/"+fn, include_days, test_post_info[0], user_ids)
    adj_lists.update(comment_adjs)

    print "Loading subscriptions"
    subscrip_adjs = load_subscriptions(user_ids, sub_ids)
    adj_lists.update(subscrip_adjs)

    print "Loading votes..."
    vote_adjs =  load_votes("/dfs/scratch0/nqe-reddit-new/vote_data/"+fn, include_days, 
            test_post_info[0], user_ids)
    adj_lists.update(vote_adjs)

    
    pickle.dump(relations, open("/dfs/scratch0/nqe-reddit-new/new_graph/rels.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL) 
    pickle.dump(adj_lists, open("/dfs/scratch0/nqe-reddit-new/new_graph/adj_lists.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL) 
    pickle.dump(test_post_info[-1], open("/dfs/scratch0/nqe-reddit-new/new_graph/post_words.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL) 

def make_train_test_edge_data(data_dir):
    print "Loading graph..."
    graph, _ = load_graph(data_dir, 10)
    print "Getting all edges..."
    edges = graph.get_all_edges()
    split_point = int(0.1*len(edges))
    val_test_edges = edges[:split_point]
    print "Getting negative samples..."
    val_test_edge_negsamples = [graph.get_negative_edge_samples(e, 100) for e in val_test_edges]
    print "Making and storing test queries."
    val_test_edge_queries = [Query(("1-chain", val_test_edges[i]), val_test_edge_negsamples[i], None, 100, keep_graph=True) for i in range(split_point)]
    val_split_point = int(0.1*len(val_test_edge_queries))
    val_queries = val_test_edge_queries[:val_split_point]
    test_queries = val_test_edge_queries[val_split_point:]
    pickle.dump([q.serialize() for q in val_queries], open(data_dir+"/val_edges.pkl", "w"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in test_queries], open(data_dir+"/test_edges.pkl", "w"), protocol=pickle.HIGHEST_PROTOCOL)

    print "Removing test edges..."
    graph.remove_edges(val_test_edges)
    print "Making and storing train queries."
    train_edges = graph.get_all_edges()
    train_queries = [Query(("1-chain", e), None, None, keep_graph=True) for e in train_edges]
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

def clean_test(data_dir):
    test_edges = pickle.load(open(data_dir + "/test_edges.pkl", "rb"))
    val_edges = pickle.load(open(data_dir + "/val_edges.pkl", "rb"))  
    deleted_edges = set([q[0][1] for q in test_edges] + [_reverse_edge(q[0][1]) for q in test_edges] + 
                [q[0][1] for q in val_edges] + [_reverse_edge(q[0][1]) for q in val_edges])

    for i in range(2,4):
        for kind in ["val", "test"]:
            if kind == "val":
                to_keep = 1000
            else:
                to_keep = 10000
            test_queries = load_queries_by_type(data_dir + "/{:s}_queries_{:d}-split.pkl".format(kind, i), keep_graph=True)
            print "Loaded", i, kind
            for query_type in test_queries:
                test_queries[query_type] = [q for q in test_queries[query_type] if len(q.get_edges().intersection(deleted_edges)) > 0]
                test_queries[query_type] = test_queries[query_type][:to_keep]
            test_queries = [q.serialize() for queries in test_queries.values() for q in queries]
            pickle.dump(test_queries, open(data_dir + "/{:s}_queries_{:d}-clean.pkl".format(kind, i), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            print "Finished", i, kind
 
def sample_new_clean(data_dir):
    graph_loader = lambda : load_graph(data_dir, 10)[0]
    sample_clean_test(graph_loader, data_dir) 

def make_train_test_query_data(data_dir):
    graph, _ = load_graph(data_dir, 10)
    print "Sampling train queries..."
    queries_2, queries_3 = parallel_sample(graph, 80, 10000, data_dir, test=False, start_ind=0)
    print "Sampling test queries..."
    t_queries_2, t_queries_3 = parallel_sample(graph, 20, 5000, data_dir, test=True)
    t_queries_2 = list(set(t_queries_2) - set(queries_2))
    t_queries_3 = list(set(t_queries_3) - set(queries_3))
    print len(t_queries_2), len(t_queries_3)
    pickle.dump([q.serialize() for q in queries_2], open(data_dir + "/train_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in queries_3], open(data_dir + "/train_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in t_queries_2[10000:]], open(data_dir + "/test_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in t_queries_3[10000:]], open(data_dir + "/test_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in t_queries_2[:10000]], open(data_dir + "/val_queries_2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in t_queries_3[:10000]], open(data_dir + "/val_queries_3.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
 #   make_train_test_edge_data("/dfs/scratch0/nqe-reddit-new/gaming_graph")
    #make_train_test_query_data("/dfs/scratch0/nqe-reddit-new/gaming_graph")
    sample_new_clean("/dfs/scratch0/nqe-reddit-new/gaming_graph/")
    #discard_negatives("/dfs/scratch0/nqe-reddit-new/gaming_graph")
    #parallel_sample(176, 6250, 16, "/dfs/scratch0/nqe-reddit/gaming_graph/")
    #random.seed(1)
    #subs_set = set([])
    #with open("/dfs/scratch0/nqe-reddit/videogame_subreddits_clean.txt", "r") as fp:
    #    for line in fp:
    #        subs_set.add(line.strip().lower())
    #build_graph("merged.csv", set([1,2,3,4,5]), subs_set=subs_set)
    """
    graph, _ = load_graph("/dfs/scratch0/nqe-reddit/tiny_graph/")
    from collections import OrderedDict
    anchor_weights = OrderedDict()
    for mode in graph.relations:
        anchor_weights[mode] = 1./len(graph.relations)
    print graph.sample_chains_byrels(2, anchor_weights, 10)
    """

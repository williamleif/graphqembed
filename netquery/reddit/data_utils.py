import numpy as np
from collections import defaultdict, Counter, OrderedDict
import cPickle as pickle
from multiprocessing import Process

from gensim.models.word2vec import Word2Vec

import spacy
import random

import torch
import torch.nn as nn
from netquery.graph import Graph



def load_comments(filename, valid_days, w2v, post_ids):
    comment_ids = {}
    replied_to = {}
    post_belongs_to = {}
    sub_belongs_to = {}
    made_by = {}

    comment_feats = []
    with open(filename) as fp:
        for line in fp:
            info = line.strip().split(",")
            if int(info[0].split("-")[-1].split()[0]) in valid_days:
                post = info[-1]
                if not (post in post_ids):
                    continue
                comment_id = len(comment_ids)
                comment_ids[info[3]] = comment_id
    print len(comment_ids), "valid comments"
    comment_feats = []
    with open(filename) as fp:
        for i, line in enumerate(fp):
            if i % 1000 == 0:
                print "Done comment", i
 
            info = line.strip().split(",")
            if not int(info[0].split("-")[-1].split()[0]) in valid_days:
                continue
            if not info[3] in comment_ids:
                continue
            comment_id = comment_ids[info[3]]
            author = info[1]
            parent = info[-2]
            post = info[-1]
            if post != parent and parent in comment_ids:
                replied_to[comment_id] = comment_ids[parent]
            if post in post_ids:
                post_belongs_to[comment_id] = post_ids[post]
            sub_belongs_to[comment_id] = info[2]
            made_by[comment_id] = author
            if len(info) > 7:
                text = ",".join(info[4:-2])
            else:
                text = info[4]
            tokenized = tokenizer(unicode(text, errors="ignore"))
            word_vecs = [w2v.wv[w.lower_] for w in tokenized if w.lower_ in w2v.wv]
            if len(word_vecs) == 0:
                word_vecs = [np.zeros(100,)]
            comment_feats.append(np.mean(word_vecs, axis=0))
    
    print len(post_belongs_to), "comment to post edges"
    return comment_ids, replied_to, post_belongs_to, sub_belongs_to, made_by, np.stack(comment_feats)

def load_posts(filename, valid_days, w2v, sub_limit=None, subs_set=None):

    post_ids = {}
    sub_belongs_to = {}
    made_by = {}
    post_type = {}
    post_feats = []
    type_map = {t:i for i,t in enumerate(["self", "link", "image", "crosspost", "video", "gif"])}
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
            post_type[post_id] = type_map[info[4]]
            text = ". ".join(info[-3:])
            tokenized = tokenizer(unicode(text, errors="ignore"))
            word_vecs = [w2v.wv[w.lower_] for w in tokenized if w.lower_ in w2v.wv]
            if len(word_vecs) == 0:
                word_vecs = [np.zeros(100,)]
            post_feats.append(np.mean(word_vecs, axis=0))

    return post_ids, sub_belongs_to, made_by, post_type, np.stack(post_feats)

def load_votes(filename, valid_days, comments, posts, user_ids):
    adj_lists = {("user", "up", "comment") : defaultdict(list),
                 ("user", "down", "comment") : defaultdict(list),
                 ("comment", "up", "user") : defaultdict(list),
                 ("comment", "down", "user") : defaultdict(list),
                 ("user", "up", "post") : defaultdict(list),
                 ("user", "down", "post") : defaultdict(list),
                 ("post", "up", "user") : defaultdict(list),
                 ("post", "down", "user") : defaultdict(list),
                 }

    with open(filename) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split(",")
            if not int(info[0].split("-")[-1].split()[0]) in valid_days or info[-1] == "clear":
                continue
            vote_type = "comment" if info[-2] == "comment" else "post" 
            if vote_type == "comment" and info[-3] in comments:
                if not info[1] in user_ids:
                    user_ids[info[1]] = len(user_ids)
                user_id = user_ids[info[1]]
                comment_id = comments[info[-3]]
                adj_lists[("user", info[-1], "comment")][user_id].append(comment_id)
                adj_lists[("comment", info[-1], "user")][comment_id].append(user_id)
            if vote_type == "post" and info[-3] in posts:
                if not info[1] in user_ids:
                    user_ids[info[1]] = len(user_ids)
                user_id = user_ids[info[1]]
                post_id = posts[info[-3]]
                adj_lists[("user", info[-1], "post")][user_id].append(post_id)
                adj_lists[("post", info[-1], "user")][post_id].append(user_id)

    return adj_lists

def load_graph(info_dir, embed_dim=16):
    print "Loading adjacency info..."
    adj_lists = pickle.load(open(info_dir + "/adj_lists.pkl"))
    relations = pickle.load(open(info_dir + "/rels.pkl"))
    print "Loading feature data.."
    post_feats = np.load(info_dir + "/post_feats.npy")
    post_feats = np.concatenate([np.zeros((1,100)), post_feats])
    comment_feats = np.load(info_dir + "/comment_feats.npy")
    comment_feats = np.concatenate([np.zeros((1,100)), comment_feats])


    num_users = len(set([id for rel, adj in adj_lists.iteritems() for id in adj if rel[0] == "user"]))
    num_communities = len(set([id for rel, adj in adj_lists.iteritems() for id in adj if rel[0] == "community"]))

    feature_modules = {
        "comment" : nn.Embedding(comment_feats.shape[0], comment_feats.shape[1]), 
        "post" : nn.Embedding(comment_feats.shape[0], comment_feats.shape[1]), 
        "user" : nn.Embedding(num_users+1, embed_dim), 
        "community" : nn.Embedding(num_communities+1, embed_dim), 
        "type" : nn.Embedding(6, embed_dim)}
    feature_modules["comment"].weight = nn.Parameter(torch.FloatTensor(comment_feats), requires_grad=False)
    feature_modules["post"].weight = nn.Parameter(torch.FloatTensor(post_feats), requires_grad=False)
    for mode in ["user", "community", "type"]:
        feature_modules[mode].weight.data.normal_(0, 1./embed_dim)
    features = lambda nodes, mode : feature_modules[mode](
            torch.autograd.Variable(torch.LongTensor(nodes)+1))
    feature_dims = {mode : embed.weight.size()[1] for mode, embed in feature_modules.iteritems()}
    graph = Graph(features, feature_dims, relations, adj_lists)
    return graph, feature_modules

def build_graph(fn, include_days, sub_limit=None, subs_set=None):
    nlp = spacy.load('en')
    global tokenizer 
    tokenizer = nlp.tokenizer

    w2v = Word2Vec.load("/dfs/scratch0/reddit/w2v_model_021518_2016_full.gsim")
    test_post_info = load_posts("/dfs/scratch0/nqe-reddit/post_data/"+fn, include_days, w2v, sub_limit=sub_limit, subs_set=subs_set)
    test_comment_info = load_comments("/dfs/scratch0/nqe-reddit/comment_data/"+fn, include_days, w2v, test_post_info[0])
    
    relations = {
            "user" : [("comment", "up"), ("comment", "down"), ("post", "up"), ("post", "down"), ("comment", "make"), ("post", "make")],
            "comment" : [("user", "up"), ("user", "down"), ("comment", "reply"), ("post", "reply"), ("user", "make")],
            "post" : [("user", "up"), ("user", "down"), ("comment", "reply"), ("community", "belong"), ("type", "is_type"), ("user", "make")],
            "community" : [("post", "belong")],
            "type" : [("post", "is_type")],
            }
    adj_lists = {}
    for mode, rels in relations.iteritems():
        for rel in rels:
            adj_lists[(mode, rel[-1], rel[0])] = defaultdict(list)

    sub_ids = {}
    user_ids = {}
    for post, sub in test_post_info[1].iteritems():
        if not sub in sub_ids:
            sub_ids[sub] = len(sub_ids)
        sub_id = sub_ids[sub]
        adj_lists[("post", "belong", "community")][post].append(sub_id)
        adj_lists[("community", "belong", "post")][sub_id].append(post)
    for post, user in test_post_info[2].iteritems():
        if not user in user_ids:
            user_ids[user] = len(user_ids)
        user_id = user_ids[user]
        adj_lists[("user", "make", "post")][user_id].append(post)
        adj_lists[("post", "make", "user")][post].append(user_id)
    for post, post_type in test_post_info[3].iteritems():
        adj_lists[("post", "is_type", "type")][post].append(post_type)
        adj_lists[("type", "is_type", "post")][post_type].append(post)
    for comment1, comment2 in test_comment_info[1].iteritems():
        adj_lists[("comment", "reply", "comment")][comment1].append(comment2)
        adj_lists[("comment", "reply", "comment")][comment2].append(comment1)
    for comment, post in test_comment_info[2].iteritems():
        adj_lists[("comment", "reply", "post")][comment].append(post)
        adj_lists[("post", "reply", "comment")][post].append(comment)
    for comment, user in test_comment_info[4].iteritems():
        if not user in user_ids:
            user_ids[user] = len(user_ids)
        user_id = user_ids[user]
        adj_lists[("comment", "make", "user")][comment].append(user_id)
        adj_lists[("user", "make", "comment")][user_id].append(comment)
    vote_adjs =  load_votes("/dfs/scratch0/nqe-reddit/vote_data/"+fn, include_days, test_comment_info[0], test_post_info[0], user_ids)
    adj_lists.update(vote_adjs)

    np.save("/dfs/scratch0/nqe-reddit/new_graph/post_feats.npy", test_post_info[-1])
    np.save("/dfs/scratch0/nqe-reddit/new_graph/comment_feats.npy", test_comment_info[-1])
    pickle.dump(relations, open("/dfs/scratch0/nqe-reddit/new_graph/rels.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL) 
    pickle.dump(adj_lists, open("/dfs/scratch0/nqe-reddit/new_graph/adj_lists.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL) 

if __name__ == "__main__":
    parallel_sample(176, 6250, 16, "/dfs/scratch0/nqe-reddit/gaming_graph/")
    """
    random.seed(1)
    subs_set = set([])
    with open("/dfs/scratch0/nqe-reddit/videogame_subreddits_clean.txt", "r") as fp:
        for line in fp:
            subs_set.add(line.strip().lower())
    build_graph("merged.csv", set([1,2,3,4,5]), subs_set=subs_set)
    graph, _ = load_graph("/dfs/scratch0/nqe-reddit/tiny_graph/")
    from collections import OrderedDict
    anchor_weights = OrderedDict()
    for mode in graph.relations:
        anchor_weights[mode] = 1./len(graph.relations)
    print graph.sample_chains_byrels(2, anchor_weights, 10)
    """

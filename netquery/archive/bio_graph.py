import os
from collections import defaultdict
from gsq.graph import make_adj_lists, Graph
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import dill as pickle

DATA_HOME = "/dfs/scratch2/MINER-BIO/data-miner/types/"
PREFIX = "/20160418/snap-tables/"

#TODO: Deal with duplicate nodes!!

def _get_data_file(relation):
    candidates = {}
    directory = DATA_HOME + relation[0] + "-" + relation[-1] + PREFIX
    if not os.path.isdir(directory):
        return None
    for filename in os.listdir(directory):
        info = filename.split(".")[0].split("-")
        if len(info) != 4:
            continue
        if "-".join(info[:3]) ==\
                "miner-"+relation[0].lower()+"-"+relation[-1].lower():
            candidates[int(info[-1])] = filename
    latest = candidates[sorted(candidates.keys())[-1]]
    return directory + "/" + latest

def _make_node_maps(valid_nodes):
    node_maps = defaultdict(dict)
    for mode, node_set in valid_nodes.iteritems():
        for i, node in enumerate(node_set):
            node_maps[mode][node] = i
    return node_maps

def _get_valid_disease_gene():
    valid_edges = {}
    with open(DATA_HOME + "/Disease-Gene/" + PREFIX + "miner-disease-gene-0-CTD_MESH-20160608.tsv") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.split("\t")
            if info[5] != "":
                valid_edges[info[0]] = info[5].split("/")[0]
    return valid_edges

def _get_valid_disease_chemical():
    valid_edges = {}
    with open(DATA_HOME + "/Disease-Chemical/" + PREFIX + "miner-disease-chemical-0-CTD_MESH-20160608.tsv") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.split("\t")
            if info[6] != "":
                valid_edges[info[0]] = info[6].split("/")[0]
    return valid_edges

def load_small(num_genes=1000, feature_dim=10):
    relations = {"Chemical" : [("Chemical", "0"), ("Gene", "0"), ("Disease", "marker"), ("Disease", "therapeutic")],
            "Disease" : [("Disease", "0"), ("Chemical", "marker"), ("Chemical", "therapeutic"), ("Gene", "marker"), ("Gene", "therapeutic")],
            "Gene" : [("Chemical", "0"), ("Disease", "marker"), ("Disease", "therapeutic"), ("Function", "0")],
            "Function" : [("Function", "0"), ("Gene", "0")]
            }

    adj_lists = make_adj_lists(relations)
    valid_nodes = defaultdict(set)
    used_nodes = defaultdict(set)
    valid_nodes["Gene"] = set(np.random.choice(range(100000), size=num_genes))
    valid_edges = {}
    for mode in relations:
        for rel in relations[mode]:
            valid_edges[(mode, rel[1], rel[0])] = lambda v : True
    disease_chem = _get_valid_disease_chemical()
    valid_edges[("Chemical", "marker", "Disease")] = lambda v : v in disease_chem and disease_chem[v] == "marker"
    valid_edges[("Disease", "marker", "Chemical")] = lambda v : v in disease_chem and disease_chem[v] == "marker"
    valid_edges[("Chemical", "therapeutic", "Disease")] = lambda v : v in disease_chem and disease_chem[v] == "therapeutic"
    valid_edges[("Disease", "therapeutic", "Chemical")] = lambda v : v in disease_chem and disease_chem[v] == "therapeutic"
    disease_gene = _get_valid_disease_gene()
    valid_edges[("Gene", "marker", "Disease")] = lambda v : v in disease_gene and disease_gene[v] == "marker"
    valid_edges[("Disease", "marker", "Gene")] = lambda v : v in disease_gene and disease_gene[v] == "marker"
    valid_edges[("Gene", "therapeutic", "Disease")] = lambda v : v in disease_gene and disease_gene[v] == "therapeutic"
    valid_edges[("Disease", "therapeutic", "Gene")] = lambda v : v in disease_gene and disease_gene[v] == "therapeutic"

    for mode in ["Gene", "Chemical", "Disease", "Function"]:
        for rel in relations[mode]:
            relation = ((mode, rel[1], rel[0]))
            filename = _get_data_file((mode, rel[0]))
            print relation
            if filename is None:
                filename = _get_data_file((rel[0], mode))
                reverse = True
            else:
                reverse = False
            with open(filename) as fp:
                for i, line in enumerate(fp):
                    if line[0] == "#":
                        continue
                    info = line.split("\t")
                    if not valid_edges[relation](info[0]):
                        continue
                    if reverse:
                        node1 = int(info[-2])
                        node2 = int(info[-1])
                    else:
                        node2 = int(info[-2])
                        node1 = int(info[-1])
                    if mode == "Gene" and node1 in valid_nodes["Gene"]:
                        adj_lists[relation][node1].append(node2)
                        valid_nodes[rel[0]].add(node2)
                        used_nodes[mode].add(node1)
                        used_nodes[rel[0]].add(node2)
                    elif node1 in valid_nodes[mode] and node2 in valid_nodes[rel[0]]:
                        adj_lists[relation][node1].append(node2)
                        used_nodes[mode].add(node1)
                        used_nodes[rel[0]].add(node2)
    node_maps = _make_node_maps(used_nodes)
    for mode, used_set in used_nodes.iteritems():
        print mode, len(used_set)
    return relations, adj_lists, node_maps

if __name__ == "__main__":
    graph_info = load_small()
    pickle.dump(graph_info, open("/dfs/scratch0/gsq/small_bio.pkl", "w"))
    #print _get_valid_disease_chemical()

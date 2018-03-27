import os
from collections import defaultdict
from netquery.graph import make_adj_lists
import random
import dill as pickle

DATA_HOME = "/dfs/scratch0/netquery/cancer_example/"

def _get_miner_data_file(relation):
    directory = DATA_HOME + relation[0] + "-" + relation[-1] + "/" 
    if not os.path.isdir(directory):
        return None
    candidates = {}
    for filename in os.listdir(directory):
        info = filename.split(".")[0].split("-")
        if len(info) != 4:
            continue
        if "-".join(info[:3]) ==\
                "miner-"+relation[0].lower()+"-"+relation[-1].lower():
            candidates[int(info[-1])] = filename
    latest = candidates[sorted(candidates.keys())[-1]]
    return directory + "/" + latest

def _get_link_data_file(relation):
    directory = DATA_HOME + relation[0] + "-" + relation[-1] + "/" 
    if not os.path.isdir(directory):
        return None
    directory += relation[1] + "_links/"
    for fn in os.listdir(directory):
        if len(fn.split("-")) == 4:
            break
    return directory + "/" + fn

def get_valid_disease_nodes(num=100):
    ids = set([])
    with open(DATA_HOME + "/disease-protein/miner-disease-protein-20170709.tsv") as fp:
        for line in fp:
            if "#" in  line:
                continue
            ids.add(int(line.split()[-2]))
    ids = list(ids)
    random.shuffle(ids)
    return set(ids[:num])

def _make_node_maps(valid_nodes):
    node_maps = defaultdict(dict)
    for mode, node_set in valid_nodes.iteritems():
        for i, node in enumerate(node_set):
            node_maps[mode][node] = i
    return node_maps

def load():
    relations = {"chemical" : [("chemical", "0"), ("protein","0"), ("disease", "0")],
                 "function" : [("function", "0"), ("protein","0"), ("disease", "0")],
                 "disease" : [("disease", "0"), ("function","0"), ("protein", "0"), ("chemical", "0")],
                 "gene" : [("protein", "0")],
                 "protein" : [("gene", "0"), ("chemical", "0"), ("function", "0"), ("disease","0")]}
    gene_gene = ["colocalization", "pathway", "genetic_interactions", "physical_interactions"]
    protein_protein = ["coexpression", "experimental", "textmining"]
    for gene_rel in gene_gene:
        relations["gene"].append(("gene", gene_rel))
    for protein_rel in protein_protein:
        relations["protein"].append(("protein", protein_rel))
    valid_diseases = get_valid_disease_nodes()
    adj_lists = make_adj_lists(relations)

    used_nodes = defaultdict(set)
    edges = defaultdict(set)
    for mode in relations.keys():
         for rel in relations[mode]:
            relation = ((mode, rel[1], rel[0]))
            print relation
            reverse = False
            if rel[1] == "0":
                filename = _get_miner_data_file((mode, rel[0]))
                if filename is None:
                    filename = _get_miner_data_file((rel[0], mode))
                    reverse = True
            else:
                filename = _get_link_data_file(relation)
            with open(filename) as fp:
                for i, line in enumerate(fp):
                    if line[0] == "#":
                        continue
                    info = line.split("\t")
                    if reverse:
                        node1 = int(info[-1])
                        node2 = int(info[-2])
                    else:
                        node2 = int(info[-1])
                        node1 = int(info[-2])
                    if (mode == "disease" and not node1 in valid_diseases)\
                            or (rel[0] == "disease" and not node2 in valid_diseases):
                        continue
                    if (node1, node2) in edges[relation]:
                        continue
                    adj_lists[relation][node1].append(node2)
                    edges[relation].add((node1, node2))
                    used_nodes[mode].add(node1)
                    used_nodes[rel[0]].add(node2)
    node_maps = _make_node_maps(used_nodes)
    for mode, used_set in used_nodes.iteritems():
        print mode, len(used_set)
    for relation, count in edges.iteritems():
        print relation, len(count)
    return relations, adj_lists, node_maps

if __name__ == "__main__":
    graph_info = load()
    pickle.dump(graph_info, open("/dfs/scratch0/netquery/cancer.pkl", "w"))
    #print _get_valid_disease_chemical()

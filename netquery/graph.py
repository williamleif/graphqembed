from collections import OrderedDict, defaultdict
import numpy as np
import random

def _reverse_relation(relation):
    return (relation[-1], relation[1], relation[0])

class Graph():
    """
    Simple container for heteregeneous graph data.
    """
    def __init__(self, features, feature_dims, relations, adj_lists):
        self.features = features
        self.feature_dims = feature_dims
        self.relations = relations
        self.adj_lists = adj_lists
        self.rel_edges = OrderedDict()
        self.full_sets = {}
        self.meta_neighs = defaultdict(dict)
        for rel, adjs in self.adj_lists.iteritems():
            full_set = set(self.adj_lists[rel].keys())
            self.full_sets[rel] = full_set

        self.edges = 0.
        self.nodes = {rel : np.array(adjs.keys()) for rel, adjs in adj_lists.iteritems()}
        self.degrees = {rel : np.array([len(adj_lists[rel][node]) 
            for node in self.nodes[rel]]) for rel in adj_lists}
        self.weights = {rel : degs/float(degs.sum()) for rel, degs in self.degrees.iteritems()}
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1,r2[1], r2[0])
                self.rel_edges[rel] = 0.
                for adj_list in self.adj_lists[rel].values():
                    self.rel_edges[rel] += len(adj_list)
                    self.edges += len(adj_list)
        self.graph_chains = defaultdict(dict)
        self.neg_chains = defaultdict(dict)
        self.graph_intersections = defaultdict(dict)
        self.neg_intersections = defaultdict(dict)

    def remove_edges(self, edge_list):
        for edge in edge_list:
            try:
                self.adj_lists[edge[-1]][edge[0]].remove(edge[1])
            except Exception:
                continue

            try:
                self.adj_lists[_reverse_relation(edge[-1])][edge[1]].remove(edge[0])
            except Exception:
                continue
        self.meta_neighs = defaultdict(dict)

    def add_edges(self, edge_list):
        for edge in edge_list:
            self.adj_lists[edge[-1]][edge[0]].append(edge[1])
            self.adj_lists[_reverse_relation(edge[-1])][edge[1]].append(edge[0])
        self.meta_neighs = defaultdict(dict)

    def get_all_edges(self, seed=0, exclude_rels=set([])):
        """
        Returns all edges in the form (relation, node1, node2)
        """
        edges = []
        np.random.seed(seed)
        for rel, adjs in self.adj_lists.iteritems():
            if rel in exclude_rels:
                continue
            for node, neighs in adjs.iteritems():
                edges.extend([(node, neigh, rel) for neigh in neighs if neigh != -1])
        random.shuffle(edges)
        return edges

    def get_all_edges_byrel(self, seed=0, exclude_rels=set([])):
        """
        Returns all edges in the form relation : list of edges
        Also returns a paired negative edge for each positive one
        """
        np.random.seed(seed)
        edges = defaultdict(list)
        neg_edges = {}
        for rel, adjs in self.adj_lists.iteritems():
            if rel in exclude_rels:
                continue
            for node, neighs in adjs.iteritems():
                edges[(rel,)].extend([(node, neigh) for neigh in neighs if neigh != -1])
        for rel, edge_list in edges.iteritems():
            random.shuffle(edge_list)
            neg_edges[(rel,)] = [(node, 
                random.choice(list(self.full_sets[_reverse_relation(rel)]-set(self.adj_lists[rel][node]))))
                for (node, neigh) in edge_list]
        return edges, neg_edges
    
    def get_metapath_neighs(self, node, rels):
        if node in self.meta_neighs[rels]:
            return self.meta_neighs[rels][node]
        current_set = [node]
        for rel in rels:
            current_set = set([neigh for n in current_set for neigh in self.adj_lists[rel][n]])
        self.meta_neighs[rels][node] = current_set
        return current_set
    
    def sample_chain_from_node(self, length, node, rel):
        rels = [rel]
        for cur_len in range(length-1):
            next_rel = random.choice(self.relations[rels[-1][-1]])
            rels.append((rels[-1][-1], next_rel[-1], next_rel[0]))

        rels = tuple(rels)
        meta_neighs = self.get_metapath_neighs(node, rels)
        rev_rel = _reverse_relation(rels[-1])
        full_set = self.full_sets[rev_rel]
        diff_set = full_set - meta_neighs
        if len(meta_neighs) == 0 or len(diff_set) == 0:
            return None, None, None
        chain = (node, random.choice(list(meta_neighs)))
        neg_chain = (node, random.choice(list(diff_set)))
        return chain, neg_chain, rels

    def sample_chain(self, length, start_mode):
        rel = random.choice(self.relations[start_mode])
        rel = (start_mode, rel[-1], rel[0])
        if len(self.adj_lists[rel]) == 0:
            return None, None, None
        node = random.choice(self.adj_lists[rel].keys())
        return self.sample_chain_from_node(length, node, rel)
                
    def sample_chains(self, length, anchor_weights, num_samples):

        anchor_counts = np.random.multinomial(num_samples, anchor_weights.values()) 
        graph_chains = defaultdict(list)
        neg_chains = defaultdict(list)
        for i, anchor_count in enumerate(anchor_counts):
            anchor_mode = anchor_weights.keys()[i]
            for _ in xrange(anchor_count):
                chain, neg_chain, rels = self.sample_chain(length, anchor_mode)
                if chain is None:
                    continue
                graph_chains[rels].append(chain)
                neg_chains[rels].append(neg_chain)
        return graph_chains, neg_chains

    def sample_polytree(self, length, target_mode, try_out=100):
        num_chains = random.randint(2,length)
        added = 0
        nodes = []
        rels_list = []

        for i in range(num_chains):
            remaining = length-added-num_chains
            if i != num_chains - 1:
                remaining = remaining if remaining == 0 else random.randint(0, remaining)
            added += remaining
            chain_len = 1 + remaining
            if i == 0:
                chain, _, rels = self.sample_chain(chain_len, target_mode)
                try_count = 0 
                while chain is None and try_count <= try_out:
                    chain, _, rels = self.sample_chain(chain_len, target_mode)
                    try_count += 1

                if chain is None:
                    return None, None, None, None, None
                target_node = chain[0]
                nodes.append(chain[-1])
                rels_list.append(tuple([_reverse_relation(rel) for rel in rels[::-1]]))
            else:
                rel = random.choice([r for r in self.relations[target_mode] 
                    if len(self.adj_lists[(target_mode, r[-1], r[0])][target_node]) > 0])
                rel = (target_mode, rel[-1], rel[0])
                chain, _, rels = self.sample_chain_from_node(chain_len, target_node, rel)
                try_count = 0
                while chain is None and try_count <= try_out:
                    chain, _, rels = self.sample_chain_from_node(chain_len, target_node, rel)
                    if chain is None:
                        try_count += 1
                    elif chain[-1] in nodes:
                        chain = None
                if chain is None:
                    return None, None, None, None, None
                nodes.append(chain[-1])
                rels_list.append(tuple([_reverse_relation(rel) for rel in rels[::-1]]))

        for i in range(len(nodes)):
            meta_neighs = self.get_metapath_neighs(nodes[i], rels_list[i])
            if i == 0:
                meta_neighs_inter = meta_neighs
                meta_neighs_union = meta_neighs
            else:
                meta_neighs_inter = meta_neighs_inter.intersection(meta_neighs)
                meta_neighs_union = meta_neighs_union.union(meta_neighs)
        hard_neg_nodes = list(meta_neighs_union-meta_neighs_inter)
        neg_nodes = list(self.full_sets[rels[0]]-meta_neighs_inter)
        if len(neg_nodes) == 0:
            return None, None, None, None, None
        if len(hard_neg_nodes) == 0:
            return None, None, None, None, None

        return target_node, random.choice(neg_nodes), random.choice(hard_neg_nodes), tuple(nodes), tuple(rels_list)


    """
    Size 3 polytree types: 
    a->b->c, d->c
    a->b->c, d->b
    a->c, d->c, b->c
    """
    def sample_polytrees(self, length, num_samples, try_out=1):
        samples = 0
        polytrees = defaultdict(list)
        neg_polytrees = defaultdict(list)
        hard_neg_polytrees = defaultdict(list)
        while samples < num_samples:
            t, n, h_n, nodes, rels = self.sample_polytree(length, random.choice(self.relations.keys()))
            if t is None:
                continue
            samples += 1
            polytrees[rels].append((t, nodes))
            neg_polytrees[rels].append((n, nodes))
            hard_neg_polytrees[rels].append((h_n, nodes))
        return polytrees, neg_polytrees, hard_neg_polytrees

    def sample_relation(self):
        rel_index = np.argmax(np.random.multinomial(1, 
            np.array(self.rel_edges.values())/self.edges))
        rel = self.rel_edges.keys()[rel_index]
        return rel

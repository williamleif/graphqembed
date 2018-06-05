from collections import OrderedDict, defaultdict
import random

def _reverse_relation(relation):
    return (relation[-1], relation[1], relation[0])

def _reverse_edge(edge):
    return (edge[-1], _reverse_relation(edge[1]), edge[0])


class Formula():

    def __init__(self, query_type, rels):
        self.query_type = query_type
        self.target_mode = rels[0][0]
        self.rels = rels
        if query_type == "1-chain" or query_type == "2-chain" or query_type == "3-chain":
            self.anchor_modes = (rels[-1][-1],)
        elif query_type == "2-inter" or query_type == "3-inter":
            self.anchor_modes = tuple([rel[-1] for rel in rels])
        elif query_type == "3-inter_chain":
            self.anchor_modes = (rels[0][-1], rels[1][-1][-1])
        elif query_type == "3-chain_inter":
            self.anchor_modes = (rels[1][0][-1], rels[1][1][-1])

    def __hash__(self):
         return hash((self.query_type, self.rels))

    def __eq__(self, other):
        return ((self.query_type, self.rels)) == ((other.query_type, other.rels))

    def __neq__(self, other):
        return ((self.query_type, self.rels)) != ((other.query_type, other.rels))

    def __str__(self):
        return self.query_type + ": " + str(self.rels)

class Query():

    def __init__(self, query_graph, neg_samples, hard_neg_samples, neg_sample_max=100, keep_graph=False):
        query_type = query_graph[0]
        if query_type == "1-chain" or query_type == "2-chain" or query_type == "3-chain":
            self.formula = Formula(query_type, tuple([query_graph[i][1] for i in range(1, len(query_graph))]))
            self.anchor_nodes = (query_graph[-1][-1],)
        elif query_type == "2-inter" or query_type == "3-inter":
            self.formula = Formula(query_type, tuple([query_graph[i][1] for i in range(1, len(query_graph))]))
            self.anchor_nodes = tuple([query_graph[i][-1] for i in range(1, len(query_graph))])
        elif query_type == "3-inter_chain":
            self.formula = Formula(query_type, (query_graph[1][1], (query_graph[2][0][1], query_graph[2][1][1])))
            self.anchor_nodes = (query_graph[1][-1], query_graph[2][-1][-1])
        elif query_type == "3-chain_inter":
            self.formula = Formula(query_type, (query_graph[1][1], (query_graph[2][0][1], query_graph[2][1][1])))
            self.anchor_nodes = (query_graph[2][0][-1], query_graph[2][1][-1])
        self.target_node = query_graph[1][0]
        if keep_graph:
            self.query_graph = query_graph
        else:
            self.query_graph = None
        if not neg_samples is None:
            self.neg_samples = list(neg_samples) if len(neg_samples) < neg_sample_max else random.sample(neg_samples, neg_sample_max)
        else:
            self.neg_samples = None
        if not hard_neg_samples is None:
            self.hard_neg_samples = list(hard_neg_samples) if len(hard_neg_samples) <= neg_sample_max else random.sample(hard_neg_samples, neg_sample_max)
        else:
            self.hard_neg_samples =  None

    def contains_edge(self, edge):
        if self.query_graph is None:
            raise Exception("Can only test edge contain if graph is kept. Reinit with keep_graph=True")
        edges =  self.query_graph[1:]
        if "inter_chain" in self.query_graph[0] or "chain_inter" in self.query_graph[0]:
            edges = (edges[0], edges[1][0], edges[1][1])
        return edge in edges or (edge[1], _reverse_relation(edge[1]), edge[0]) in edges

    def get_edges(self):
        if self.query_graph is None:
            raise Exception("Can only test edge contain if graph is kept. Reinit with keep_graph=True")
        edges =  self.query_graph[1:]
        if "inter_chain" in self.query_graph[0] or "chain_inter" in self.query_graph[0]:
            edges = (edges[0], edges[1][0], edges[1][1])
        return set(edges).union(set([(e[-1], _reverse_relation(e[1]), e[0]) for e in edges]))

    def __hash__(self):
         return hash((self.formula, self.target_node, self.anchor_nodes))

    def __eq__(self, other):
        return (self.formula, self.target_node, self.anchor_nodes) == (other.formula, other.target_node, other.anchor_nodes)

    def __neq__(self, other):
        return self.__hash__() != other.__hash__()

    def serialize(self):
        if self.query_graph is None:
            raise Exception("Cannot serialize query loaded with query graph!")
        return (self.query_graph, self.neg_samples, self.hard_neg_samples)

    @staticmethod
    def deserialize(serial_info, keep_graph=False):
        return Query(serial_info[0], serial_info[1], serial_info[2], None if serial_info[1] is None else len(serial_info[1]), keep_graph=keep_graph)



class Graph():
    """
    Simple container for heteregeneous graph data.
    """
    def __init__(self, features, feature_dims, relations, adj_lists):
        self.features = features
        self.feature_dims = feature_dims
        self.relations = relations
        self.adj_lists = adj_lists
        self.full_sets = defaultdict(set)
        self.full_lists = {}
        self.meta_neighs = defaultdict(dict)
        for rel, adjs in self.adj_lists.iteritems():
            full_set = set(self.adj_lists[rel].keys())
            self.full_sets[rel[0]] = self.full_sets[rel[0]].union(full_set)
        for mode, full_set in self.full_sets.iteritems():
            self.full_lists[mode] = list(full_set)
        self._cache_edge_counts()
        self._make_flat_adj_lists()

    def _make_flat_adj_lists(self):
        self.flat_adj_lists = defaultdict(lambda : defaultdict(list))
        for rel, adjs in self.adj_lists.iteritems():
            for node, neighs in adjs.iteritems():
                self.flat_adj_lists[rel[0]][node].extend([(rel, neigh) for neigh in neighs])

    def _cache_edge_counts(self):
        self.edges = 0.
        self.rel_edges = {}
        for r1 in self.relations:
            for r2 in self.relations[r1]:
                rel = (r1,r2[1], r2[0])
                self.rel_edges[rel] = 0.
                for adj_list in self.adj_lists[rel].values():
                    self.rel_edges[rel] += len(adj_list)
                    self.edges += 1.
        self.rel_weights = OrderedDict()
        self.mode_edges = defaultdict(float)
        self.mode_weights = OrderedDict()
        for rel, edge_count in self.rel_edges.iteritems():
            self.rel_weights[rel] = edge_count / self.edges
            self.mode_edges[rel[0]] += edge_count
        for mode, edge_count in self.mode_edges.iteritems():
            self.mode_weights[mode] = edge_count / self.edges

    def remove_edges(self, edge_list):
        for edge in edge_list:
            try:
                self.adj_lists[edge[1]][edge[0]].remove(edge[-1])
            except Exception:
                continue

            try:
                self.adj_lists[_reverse_relation(edge[1])][edge[-1]].remove(edge[0])
            except Exception:
                continue
        self.meta_neighs = defaultdict(dict)
        self._cache_edge_counts()
        self._make_flat_adj_lists()

    def get_all_edges(self, seed=0, exclude_rels=set([])):
        """
        Returns all edges in the form (node1, relation, node2)
        """
        edges = []
        random.seed(seed)
        for rel, adjs in self.adj_lists.iteritems():
            if rel in exclude_rels:
                continue
            for node, neighs in adjs.iteritems():
                edges.extend([(node, rel, neigh) for neigh in neighs if neigh != -1])
        random.shuffle(edges)
        return edges

    def get_all_edges_byrel(self, seed=0, 
            exclude_rels=set([])):
        random.seed(seed)
        edges = defaultdict(list)
        for rel, adjs in self.adj_lists.iteritems():
            if rel in exclude_rels:
                continue
            for node, neighs in adjs.iteritems():
                edges[(rel,)].extend([(node, neigh) for neigh in neighs if neigh != -1])

    def get_negative_edge_samples(self, edge, num, rejection_sample=True):
        if rejection_sample:
            neg_nodes = set([])
            counter = 0
            while len(neg_nodes) < num:
                neg_node = random.choice(self.full_lists[edge[1][0]])
                if not neg_node in self.adj_lists[_reverse_relation(edge[1])][edge[2]]:
                    neg_nodes.add(neg_node)
                counter += 1
                if counter > 100*num:
                    return self.get_negative_edge_samples(edge, num, rejection_sample=False)
        else:
            neg_nodes = self.full_sets[edge[1][0]] - self.adj_lists[_reverse_relation(edge[1])][edge[2]]
        neg_nodes = list(neg_nodes) if len(neg_nodes) <= num else random.sample(list(neg_nodes), num)
        return neg_nodes

    def sample_test_queries(self, train_graph, q_types, samples_per_type, neg_sample_max, verbose=True):
        queries = []
        for q_type in q_types:
            sampled = 0
            while sampled < samples_per_type:
                q = self.sample_query_subgraph_bytype(q_type)
                if q is None or not train_graph._is_negative(q, q[1][0], False):
                    continue
                negs, hard_negs = self.get_negative_samples(q)
                if negs is None or ("inter" in q[0] and hard_negs is None):
                    continue
                query = Query(q, negs, hard_negs, neg_sample_max=neg_sample_max, keep_graph=True)
                queries.append(query)
                sampled += 1
                if sampled % 1000 == 0 and verbose:
                    print "Sampled", sampled
        return queries

    def sample_queries(self, arity, num_samples, neg_sample_max, verbose=True):
        sampled = 0
        queries = []
        while sampled < num_samples:
            q = self.sample_query_subgraph(arity)
            if q is None:
                continue
            negs, hard_negs = self.get_negative_samples(q)
            if negs is None or ("inter" in q[0] and hard_negs is None):
                continue
            query = Query(q, negs, hard_negs, neg_sample_max=neg_sample_max, keep_graph=True)
            queries.append(query)
            sampled += 1
            if sampled % 1000 == 0 and verbose:
                print "Sampled", sampled
        return queries


    def get_negative_samples(self, query):
        if query[0] == "3-chain" or query[0] == "2-chain":
            edges = query[1:]
            rels = [_reverse_relation(edge[1]) for edge in edges[::-1]]
            meta_neighs = self.get_metapath_neighs(query[-1][-1], tuple(rels))
            negative_samples = self.full_sets[query[1][1][0]] - meta_neighs
            if len(negative_samples) == 0:
                return None, None
            else:
                return negative_samples, None
        elif query[0] == "2-inter" or query[0] == "3-inter":
            rel_1 = _reverse_relation(query[1][1])
            union_neighs = self.adj_lists[rel_1][query[1][-1]]
            inter_neighs = self.adj_lists[rel_1][query[1][-1]]
            for i in range(2,len(query)):
                rel = _reverse_relation(query[i][1])
                union_neighs = union_neighs.union(self.adj_lists[rel][query[i][-1]])
                inter_neighs = inter_neighs.intersection(self.adj_lists[rel][query[i][-1]])
            neg_samples = self.full_sets[query[1][1][0]] - inter_neighs
            hard_neg_samples = union_neighs - inter_neighs
            if len(neg_samples) == 0 or len(hard_neg_samples) == 0:
                return None, None
            return neg_samples, hard_neg_samples
        elif query[0] == "3-inter_chain":
            rel_1 = _reverse_relation(query[1][1])
            union_neighs = self.adj_lists[rel_1][query[1][-1]]
            inter_neighs = self.adj_lists[rel_1][query[1][-1]]
            chain_rels = [_reverse_relation(edge[1]) for edge in query[2][::-1]]
            chain_neighs = self.get_metapath_neighs(query[2][-1][-1], tuple(chain_rels))
            union_neighs = union_neighs.union(chain_neighs)
            inter_neighs = inter_neighs.intersection(chain_neighs)
            neg_samples = self.full_sets[query[1][1][0]] - inter_neighs
            hard_neg_samples = union_neighs - inter_neighs
            if len(neg_samples) == 0 or len(hard_neg_samples) == 0:
                return None, None
            return neg_samples, hard_neg_samples
        elif query[0] == "3-chain_inter":
            inter_rel_1 = _reverse_relation(query[-1][0][1])
            inter_neighs_1 = self.adj_lists[inter_rel_1][query[-1][0][-1]]
            inter_rel_2 = _reverse_relation(query[-1][1][1])
            inter_neighs_2 = self.adj_lists[inter_rel_2][query[-1][1][-1]]
            
            inter_neighs = inter_neighs_1.intersection(inter_neighs_2)
            union_neighs = inter_neighs_1.union(inter_neighs_2)
            rel = _reverse_relation(query[1][1])
            pos_nodes = set([n for neigh in inter_neighs for n in self.adj_lists[rel][neigh]]) 
            union_pos_nodes = set([n for neigh in union_neighs for n in self.adj_lists[rel][neigh]]) 
            neg_samples = self.full_sets[query[1][1][0]] - pos_nodes
            hard_neg_samples = union_pos_nodes - pos_nodes
            if len(neg_samples) == 0 or len(hard_neg_samples) == 0:
                return None, None
            return neg_samples, hard_neg_samples

    def sample_edge(self, node, mode):
        rel, neigh = random.choice(self.flat_adj_lists[mode][node])
        edge = (node, rel, neigh)
        return edge

    def sample_query_subgraph_bytype(self, q_type, start_node=None):
        if start_node is None:
            start_rel = random.choice(self.adj_lists.keys())
            node = random.choice(self.adj_lists[start_rel].keys())
            mode = start_rel[0]
        else:
            node, mode = start_node

        if q_type[0] == "3":
            if q_type == "3-chain" or q_type == "3-chain_inter":
                num_edges = 1
            elif q_type == "3-inter_chain":
                num_edges = 2
            elif q_type == "3-inter":
                num_edges = 3
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                next_query = self.sample_query_subgraph_bytype(
                    "2-chain" if q_type == "3-chain" else "2-inter", start_node=(neigh, rel[0]))
                if next_query is None:
                    return None
                if next_query[0] == "2-chain":
                    return ("3-chain", edge, next_query[1], next_query[2])
                else:
                    return ("3-chain_inter", edge, (next_query[1], next_query[2]))
            elif num_edges == 2:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("3-inter_chain", edge_1, (edge_2, self.sample_edge(neigh_2, rel_2[-1])))
            elif num_edges == 3:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (rel_1, neigh_1) == (rel_2, neigh_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                neigh_3 = neigh_1
                rel_3 = rel_1
                while ((rel_1, neigh_1) == (rel_3, neigh_3)) or ((rel_2, neigh_2) == (rel_3, neigh_3)):
                    rel_3, neigh_3 = random.choice(self.flat_adj_lists[mode][node])
                edge_3 = (node, rel_3, neigh_3)
                return ("3-inter", edge_1, edge_2, edge_3)

        if q_type[0] == "2":
            num_edges = 1 if q_type == "2-chain" else 2
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                return ("2-chain", edge, self.sample_edge(neigh, rel[-1]))
            elif num_edges == 2:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("2-inter", edge_1, edge_2)


    def sample_query_subgraph(self, arity, start_node=None):
        if start_node is None:
            start_rel = random.choice(self.adj_lists.keys())
            node = random.choice(self.adj_lists[start_rel].keys())
            mode = start_rel[0]
        else:
            node, mode = start_node
        if arity > 3 or arity < 2:
            raise Exception("Only arity of at most 3 is supported for queries")

        if arity == 3:
            # 1/2 prob of 1 edge, 1/4 prob of 2, 1/4 prob of 3
            num_edges = random.choice([1,1,2,3])
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                next_query = self.sample_query_subgraph(2, start_node=(neigh, rel[0]))
                if next_query is None:
                    return None
                if next_query[0] == "2-chain":
                    return ("3-chain", edge, next_query[1], next_query[2])
                else:
                    return ("3-chain_inter", edge, (next_query[1], next_query[2]))
            elif num_edges == 2:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("3-inter_chain", edge_1, (edge_2, self.sample_edge(neigh_2, rel_2[-1])))
            elif num_edges == 3:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (rel_1, neigh_1) == (rel_2, neigh_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                neigh_3 = neigh_1
                rel_3 = rel_1
                while ((rel_1, neigh_1) == (rel_3, neigh_3)) or ((rel_2, neigh_2) == (rel_3, neigh_3)):
                    rel_3, neigh_3 = random.choice(self.flat_adj_lists[mode][node])
                edge_3 = (node, rel_3, neigh_3)
                return ("3-inter", edge_1, edge_2, edge_3)

        if arity == 2:
            num_edges = random.choice([1,2])
            if num_edges > len(self.flat_adj_lists[mode][node]):
                return None
            if num_edges == 1:
                rel, neigh = random.choice(self.flat_adj_lists[mode][node])
                edge = (node, rel, neigh)
                return ("2-chain", edge, self.sample_edge(neigh, rel[-1]))
            elif num_edges == 2:
                rel_1, neigh_1 = random.choice(self.flat_adj_lists[mode][node])
                edge_1 = (node, rel_1, neigh_1)
                neigh_2 = neigh_1
                rel_2 = rel_1
                while (neigh_1, rel_1) == (neigh_2, rel_2):
                    rel_2, neigh_2 = random.choice(self.flat_adj_lists[mode][node])
                edge_2 = (node, rel_2, neigh_2)
                return ("2-inter", edge_1, edge_2)

    def get_metapath_neighs(self, node, rels):
        if node in self.meta_neighs[rels]:
            return self.meta_neighs[rels][node]
        current_set = [node]
        for rel in rels:
            current_set = set([neigh for n in current_set for neigh in self.adj_lists[rel][n]])
        self.meta_neighs[rels][node] = current_set
        return current_set

    ## TESTING CODE

    def _check_edge(self, query, i):
        return query[i][-1] in self.adj_lists[query[i][1]][query[i][0]]

    def _is_subgraph(self, query, verbose):
        if query[0] == "3-chain":
            for i in range(3):
                if not self._check_edge(query, i+1):
                    raise Exception(str(query))
            if not (query[1][-1] == query[2][0] and query[2][-1] == query[3][0]):
                raise Exception(str(query))
        if query[0] == "2-chain":
            for i in range(2):
                if not self._check_edge(query, i+1):
                    raise Exception(str(query))
            if not query[1][-1] == query[2][0]:
                raise Exception(str(query))
        if query[0] == "2-inter":
            for i in range(2):
                if not self._check_edge(query, i+1):
                    raise Exception(str(query))
            if not query[1][0] == query[2][0]:
                raise Exception(str(query))
        if query[0] == "3-inter":
            for i in range(3):
                if not self._check_edge(query, i+1):
                    raise Exception(str(query))
            if not (query[1][0] == query[2][0] and query[2][0] == query[3][0]):
                raise Exception(str(query))
        if query[0] == "3-inter_chain":
            if not (self._check_edge(query, 1) and self._check_edge(query[2], 0) and self._check_edge(query[2], 1)):
                raise Exception(str(query))
            if not (query[1][0] == query[2][0][0] and query[2][0][-1] == query[2][1][0]):
                raise Exception(str(query))
        if query[0] == "3-chain_inter":
            if not (self._check_edge(query, 1) and self._check_edge(query[2], 0) and self._check_edge(query[2], 1)):
                raise Exception(str(query))
            if not (query[1][-1] == query[2][0][0] and query[2][0][0] == query[2][1][0]):
                raise Exception(str(query))
        return True

    def _is_negative(self, query, neg_node, is_hard):
        if query[0] == "2-chain":
            query = (query[0], (neg_node, query[1][1], query[1][2]), query[2])
            if query[2][-1] in self.get_metapath_neighs(query[1][0], (query[1][1], query[2][1])):
                return False
        if query[0] == "3-chain":
            query = (query[0], (neg_node, query[1][1], query[1][2]), query[2], query[3])
            if query[3][-1] in self.get_metapath_neighs(query[1][0], (query[1][1], query[2][1], query[3][1])):
                return False
        if query[0] == "2-inter":
            query = (query[0], (neg_node, query[1][1], query[1][2]), (neg_node, query[2][1], query[2][2]))
            if not is_hard:
                if self._check_edge(query, 1) and self._check_edge(query, 2):
                    return False
            else:
                if (self._check_edge(query, 1) and self._check_edge(query, 2)) or not (self._check_edge(query, 1) or self._check_edge(query, 2)):
                    return False
        if query[0] == "3-inter":
            query = (query[0], (neg_node, query[1][1], query[1][2]), (neg_node, query[2][1], query[2][2]), (neg_node, query[3][1], query[3][2]))
            if not is_hard:
                if self._check_edge(query, 1) and self._check_edge(query, 2) and self._check_edge(query, 3):
                    return False
            else:
                if (self._check_edge(query, 1) and self._check_edge(query, 2) and self._check_edge(query, 3))\
                        or not (self._check_edge(query, 1) or self._check_edge(query, 2) or self._check_edge(query, 3)):
                    return False
        if query[0] == "3-inter_chain":
            query = (query[0], (neg_node, query[1][1], query[1][2]), ((neg_node, query[2][0][1], query[2][0][2]), query[2][1]))
            meta_check = lambda : query[2][-1][-1] in self.get_metapath_neighs(query[1][0], (query[2][0][1], query[2][1][1]))
            neigh_check = lambda : self._check_edge(query, 1)
            if not is_hard:
                if meta_check() and neigh_check():
                    return False
            else:
                if (meta_check() and neigh_check()) or not (meta_check() or neigh_check()):
                    return False
        if query[0] == "3-chain_inter":
            query = (query[0], (neg_node, query[1][1], query[1][2]), query[2])
            target_neigh = self.adj_lists[query[1][1]][neg_node]
            neigh_1 = self.adj_lists[_reverse_relation(query[2][0][1])][query[2][0][-1]]
            neigh_2 = self.adj_lists[_reverse_relation(query[2][1][1])][query[2][1][-1]]
            if not is_hard:
                if target_neigh in neigh_1.intersection(neigh_2):
                    return False
            else:
                if target_neigh in neigh_1.intersection(neigh_2) and not target_neigh in neigh_1.union(neigh_2):
                    return False
        return True

            

    def _run_test(self, num_samples=1000):
        for i in range(num_samples):
            q = self.sample_query_subgraph(2)
            if q is None:
                continue
            self._is_subgraph(q, True)
            negs, hard_negs = self.get_negative_samples(q)
            if not negs is None:
                for n in negs:
                    self._is_negative(q, n, False)
            if not hard_negs is None:
                for n in hard_negs:
                    self._is_negative(q, n, True)
            q = self.sample_query_subgraph(3)
            if q is None:
                continue
            self._is_subgraph(q, True)
            negs, hard_negs = self.get_negative_samples(q)
            if not negs is None:
                for n in negs:
                    self._is_negative(q, n, False)
            if not hard_negs is None:
                for n in hard_negs:
                    self._is_negative(q, n, True)
        return True


    """
    TO DELETE? 
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
        sampled = 0
        graph_chains = defaultdict(list)
        neg_chains = defaultdict(list)
        while sampled < num_samples: 
            anchor_mode = anchor_weights.keys()[np.argmax(np.random.multinomial(1, anchor_weights.values()))]
            chain, neg_chain, rels = self.sample_chain(length, anchor_mode)
            if chain is None:
                continue
            graph_chains[rels].append(chain)
            neg_chains[rels].append(neg_chain)
            sampled += 1
        return graph_chains, neg_chains


    def sample_polytree_rootinter(self, length, target_mode, try_out=100):
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

        return target_node, neg_nodes, hard_neg_nodes, tuple(nodes), tuple(rels_list)


    def sample_polytrees_parallel(self, length, thread_samples, threads, try_out=100):
        pool = Pool(threads)
        sample_func = partial(self.sample_polytree, length)
        sizes = [thread_samples for _ in range(threads)]
        results = pool.map(sample_func, sizes)
        polytrees = {}
        neg_polytrees = {}
        hard_neg_polytrees = {}
        for p, n, h in results: 
            polytrees.update(p)
            neg_polytrees.update(n)
            hard_neg_polytrees.updarte(h)
        return polytrees, neg_polytrees, hard_neg_polytrees
        
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

    """

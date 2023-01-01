import time
import json
import networkx as nx
from scipy.stats import kendalltau

    
def get_query_results_ranking(q, k=None):

    ans = {}

    for result in q.results:
        tuple_id = result.tuple_id

        ranked_facts = sorted(result.facts, key=lambda fact: (-fact.shapley_value, fact.tuple_id))
        if k is not None:
            ranked_facts = ranked_facts[:k]
        ans[tuple_id] = ranked_facts
            
    return ans

def compute_edge_weight(r1, r2):
    r1_array = [x.tuple_id for x in r1]
    r2_array = [x.tuple_id for x in r2]

    r1_set = set(x.tuple_id for x in r1)
    r2_set = set(x.tuple_id for x in r2)
    u = r1_set.union(r2_set)
    n = len(u)

    # add missing tuples to each list
    # kendall tau expects lists of equal length
    for x in sorted(u - r1_set):
        r1_array.append(x)
    for x in sorted(u - r2_set):
        r2_array.append(x)
    
    correlation, _ = kendalltau(r1_array, r2_array)
    
    # Convert from kendall tau correlation to normalized distance
    return (correlation + 1)/2


def compute_ranking_based_similarity(rankings1, rankings2):
    B = nx.Graph()
    B.add_nodes_from(["0_" + o[0] for o in rankings1.keys()], bipartite=0)
    B.add_nodes_from(["1_" + o[0] for o in rankings2.keys()], bipartite=1)

    for o1 in rankings1:
        for o2 in rankings2:
            w = compute_edge_weight(rankings1[o1], rankings2[o2])
            if w > 0:
                B.add_edge("0_" + o1[0], 
                           "1_" + o2[0], 
                           weight=w)

    matching = nx.max_weight_matching(B)

    if len(B.nodes) == len(matching):
        return 0
    return sum(B[e[0]][e[1]]["weight"] for e in matching) / (len(B.nodes)-len(matching))



class Similarity:
    def __init__(self, cache_filename=None):
        self.cache = dict()
        if cache_filename:
            self.load_cache(cache_filename)

    def save_cache(self, filename):
        to_save = {f"{k[0]}__{k[1]}":v for k,v in self.cache.items()}
        with open(filename, "w") as f:
            json.dump(to_save, f)

    def load_cache(self, filename):
        with open(filename, "r") as f:
            to_load = json.load(f)
        for k, v in to_load.items():
            k = tuple(k.split("__"))
            self.cache[k] = v

    def clear_cache(self):
        self.cache.clear()

    def similarity(self, q1, q2, k=None):
        cache_key = tuple(sorted([q1.query_name, q2.query_name]))
        if cache_key in self.cache:
            return self.cache[cache_key]

        rankings1 = get_query_results_ranking(q1, k=k)
        rankings2 = get_query_results_ranking(q2, k=k)

        similarity = compute_ranking_based_similarity(rankings1, rankings2)
        self.cache[cache_key] = similarity
        return similarity
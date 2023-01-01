import json


def jaccard_similarity(A, B):
    # Find intersection of two sets
    nominator = A.intersection(B)

    # Find union of two sets
    denominator = A.union(B)

    similarity = 0
    if len(denominator) > 0:
        # Take the ratio of sizes
        similarity = len(nominator) / len(denominator)
    return similarity



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

    def similarity(self, q1, q2, tuples=None):
        cache_key = tuple(sorted([q1.query_name, q2.query_name]))
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        q1_results = set([tuple(tuples[result.tuple_id]) for result in q1.results])
        q2_results = set([tuple(tuples[result.tuple_id]) for result in q2.results])

        similarity = jaccard_similarity(q1_results, q2_results)
        self.cache[cache_key] = similarity
        return similarity



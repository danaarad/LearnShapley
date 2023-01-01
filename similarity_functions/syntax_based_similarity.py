import re
import json
import mo_sql_parsing
    

def parse_conditions_from_query(query):
    selection_conditions = set()
    equi_join_conditions = set()

    from_clause = json.dumps(query["from"])
    if "and" in query["where"]:
        conds = query["where"]["and"]
    else:
        conds = [query["where"]]

    for cond in conds:
        cond_is_join = False
        if "eq" in cond:
            try:
                possible_col1 = cond["eq"][0]
                split = possible_col1.split(".")
                possible_table1 = split[0] if len(split) == 2 else None

                possible_col2 = cond["eq"][1]
                split = possible_col2.split(".")
                possible_table2 = split[0] if len(split) == 2 else None

                if (possible_table1 and possible_table2 and
                            possible_table1 in from_clause and
                            possible_table2 in from_clause):
                    cond_is_join = True
                    cond["eq"].sort()
                    equi_join_conditions.add(json.dumps(cond))
            except AttributeError:
                pass
        if not cond_is_join:
            selection_conditions.add(json.dumps(cond))
    return selection_conditions, equi_join_conditions


def get_projection_operation(query):
    return json.dumps(query["select"])


def get_query_operations(query):
    projection_operation = get_projection_operation(query)
    selection_operations, equi_join_operations = parse_conditions_from_query(query)

    operations = {projection_operation}
    operations.update(selection_operations)
    operations.update(equi_join_operations)

    return operations


def jaccard_similarity(A, B):
    # Find intersection of two sets
    nominator = A.intersection(B)

    # Find union of two sets
    denominator = A.union(B)

    similarity = None
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

    def similarity(self, q1, q2):
        cache_key = tuple(sorted([q1.query_name, q2.query_name]))
        if cache_key in self.cache:
            return self.cache[cache_key]

        q1 = mo_sql_parsing.parse(q1.sql)
        q2 = mo_sql_parsing.parse(q2.sql)

        q1_operations = get_query_operations(q1)
        q2_operations = get_query_operations(q2)

        similarity = jaccard_similarity(q1_operations, q2_operations)
        self.cache[cache_key] = similarity
        return similarity
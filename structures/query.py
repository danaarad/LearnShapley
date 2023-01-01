

class Query:
    def __init__(self, query_name, sql, results, load_results=True):
        self.query_name = query_name
        self.sql = sql
        if load_results:
            self.results = self.load_results(results)
        else:
            self.results = []

    def load_results(self, raw_results):
        results = []
        for raw_result in raw_results:
            facts = [Fact(**raw_fact) for raw_fact in raw_result["facts"]]
            result = Result(raw_result["tuple_id"], facts)
            results.append(result)
        return results


class Tuple:
    def __init__(self, tuple_id):
        self.tuple_id = tuple_id
        self.index = None

    def __repr__(self):
        return f"Tuple {self.tuple_id}"

class Result(Tuple):
    def __init__(self, tuple_id, facts):
        Tuple.__init__(self, tuple_id)
        self.facts = facts


class Fact(Tuple):
    def __init__(self, tuple_id, shapley_value):
        Tuple.__init__(self, tuple_id)
        self.shapley_value = shapley_value



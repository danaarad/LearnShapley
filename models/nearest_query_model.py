# nearest_queries_cache = dict()


class NearestQueries:
    def __init__(self, k, n_neighbors, query_log, tuples_to_values, similarity_metric, similarity_metric_name=""):
        self.k = k
        self.n_neighbors = n_neighbors
        self.query_log = query_log
        self.tuples_to_values = tuples_to_values
        self.queries = {q.query_name: q for q in self.query_log}
        self.similarity_metric = similarity_metric
        self.similarity_metric_name = similarity_metric_name
        self.avg_num_neighbors = 0
        self.count_with_n_neighbors = 0

    def get_top_k(self, query, output):
        """
        Naive KNN implementation

        :param query: Query object
        :param output: Tuple, the output tuple to get top k w.r.t.
        :return: top k facts with highest contribution (Shapley value) w.r.t output and query.
        """
        if query.query_name not in self.queries:
            self.queries[query.query_name] = query

        # calculate similarity for queries which contain the given output
        # in their results
        output_value = self.tuples_to_values[output.tuple_id]
        if self.similarity_metric_name == "rank":
            similarities = [
                dict(other_query=other_query, similarity=self.similarity_metric(query, other_query, k=20))
                for other_query in self.query_log
            ]
            similarities = [sim for sim in similarities if sim["similarity"] is not None]
        elif self.similarity_metric_name == "witness":
            similarities = [
                dict(other_query=other_query, similarity=self.similarity_metric(query, other_query, self.tuples_to_values))
                for other_query in self.query_log
            ]
        else:
            similarities = [
                dict(other_query=other_query, similarity=self.similarity_metric(query, other_query))
                for other_query in self.query_log
            ]
        similarities.sort(key=lambda x: (-x["similarity"], x["other_query"].query_name))
        nearest_queries = [d["other_query"] for d in similarities]

        # sum Shapley values of the top k facts form each of the n queries
        n_nearest_top_k = dict()
        n_nearest_top_k_count = dict()
        neighbors_counter = 0
        for nearest_query in nearest_queries:
            for result in nearest_query.results:
                # we checked that such value exists
                result_value = self.tuples_to_values[result.tuple_id]
                if result_value == output_value:
                    # print("nearest query:")
                    # print("\t"+nearest_query.sql)
                    neighbors_counter += 1
                    result.facts.sort(key=lambda x: (-x.shapley_value, x.tuple_id))
                    top_k_for_result = result.facts[:self.k]
                    for tup in top_k_for_result:
                        if tup.tuple_id not in n_nearest_top_k:
                            n_nearest_top_k[tup.tuple_id] = 0
                            n_nearest_top_k_count[tup.tuple_id] = 0
                        n_nearest_top_k[tup.tuple_id] += tup.shapley_value
                        n_nearest_top_k_count[tup.tuple_id] += 1
                    break

            if neighbors_counter == self.n_neighbors:
                self.count_with_n_neighbors += 1
                break
        
        # normalize to avg contribution
        self.avg_num_neighbors += neighbors_counter
        for tup, shap_sum in n_nearest_top_k.items():
            num_tuples = n_nearest_top_k_count[tup]
            n_nearest_top_k[tup] = shap_sum / num_tuples
            # n_nearest_top_k[tup] = shap_sum

        # get overall top k facts
        n_nearest_top_k = list(n_nearest_top_k.items())
        n_nearest_top_k.sort(key=lambda x: (-x[1], x[0]))
        top_k_tuple_ids = [t[0] for t in n_nearest_top_k[:self.k]]
        top_k_scores = [t[1] for t in n_nearest_top_k[:self.k]]
        return top_k_tuple_ids, top_k_scores

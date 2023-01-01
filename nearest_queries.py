import time
from tqdm import tqdm
import similarity_functions.syntax_based_similarity as syntax_similarity
# import similarity_functions.witness_based_similarity as witness_similarity
# import similarity_functions.rank_based_similarity as rank_similarity
from models.nearest_query_model import NearestQueries
from models.utils import get_ndcg_from_scores, get_p_at_k_from_scores, get_p_at_k_with_ties_from_scores
from structures.query_log import load_query_data, get_tuples_to_values_mapping


SIMILARITY_NAME = "syntax"
SPLIT = "test"
PERCENT = 100



def run_nq(similarity_metric, similarity_name,
                        train_queries, dev_queries,
                        tuples,
                        k_values, n_values,
                        metrics,
                        save_path, percent, split):

    for n in n_values:
        for k in k_values:
            print(f"{similarity_name}-NearestQueries. processing k={k}, n={n} percent={percent} {split}")
            
            model = NearestQueries(
                k=k, n_neighbors=n,
                query_log=train_queries,
                tuples_to_values=tuples,
                similarity_metric=similarity_metric,
                similarity_metric_name=similarity_name
            )
            total_samples = 0
            avg_query_scores = {}
            times = []
            for query in tqdm(dev_queries):
                query_scores = {}
                for result in query.results:
                    t0 = time.time()
                    if len(result.facts) < k:
                        continue
                    
                    # predicted rank is constructed in the following way:
                    #   1. take the gold facts
                    #   2. rank facts with non zero predicted contribution
                    #   3. facts with predicted zero contributions are ordered at the end by tuple_id (tie breaker)

                    result.facts.sort(key=lambda x: (-x.shapley_value, x.tuple_id))
                    gold_scores = [(f.tuple_id, f.shapley_value) for f in result.facts[:k]]

                    predicted_ids, predicted_scores = model.get_top_k(query, result)
                    sorted_predictions = list(zip(predicted_ids, predicted_scores))
                    sorted_predictions.sort(key=lambda x: (-x[1], x[0]))

                    for eval_metric_name, eval_metric, metric_params in metrics:
                        score = eval_metric(sorted_predictions, gold_scores, use_lineage=True, **metric_params)
                        query_scores[eval_metric_name] = query_scores.get(eval_metric_name, 0) + score
                    total_samples += 1
                    t1 = time.time()
                    t = t1-t0
                    times.append(t)

                for eval_metric_name, avg_metric in query_scores.items():
                    query_scores[eval_metric_name] = avg_metric / len(query.results)
                    avg_query_scores[eval_metric_name] = avg_query_scores.get(eval_metric_name, 0) + query_scores[eval_metric_name]

            print(f"DONE. avg time per query + output tuple {sum(times)/len(times)}, max time for query + output tuple: {max(times)}")
            for eval_metric_name, avg_metric in avg_query_scores.items():
                avg_query_scores[eval_metric_name] = avg_query_scores[eval_metric_name] / len(dev_queries)
                print(f"avg {eval_metric_name} for k={k}, n={n} is {avg_query_scores[eval_metric_name]}")

            with open(f"{save_path}/{similarity_name}_{percent}_{split}.txt", "ab") as f:
                metric_names = list(avg_query_scores.keys())
                metric_names.sort()
                line = ",".join([str(round(v,3)) for v in avg_query_scores.values()])
                print(line)
                f.write(f"{k},{n},{line},{model.count_with_n_neighbors},{total_samples}\n".encode())


def main():
    for dataset in ["academic", "imdb"]:
        if dataset == "imdb":
            data_path = "./data/"
        if dataset == "academic":
            data_path = "./data/academic/"

        if PERCENT == 100:
            train_queries_file = f"{data_path}/train_5000_results.json"
        else:
            train_queries_file = f"{data_path}/train_5000_results_{PERCENT}.json"
        train_queries = load_query_data(train_queries_file)

        # k_values = [3, 4, 5, 6, 7, 8, 9, 10, 15, 30, 45, 60]
        # n_values = [1, 3]

        k_values = [10]
        n_values = [3]

        tuples = get_tuples_to_values_mapping(f"{data_path}/train_tuples.json")
        dev_tuples = get_tuples_to_values_mapping(f"{data_path}/dev_tuples.json")
        tuples.update(dev_tuples)
        test_tuples = get_tuples_to_values_mapping(f"{data_path}/test_tuples.json")
        tuples.update(test_tuples)

        cache_file = f"./similarity_functions/similarity_cache/{dataset}_{SIMILARITY_NAME}_similarity_cache.json"

        metrics = [
            ("ndcg", get_ndcg_from_scores, dict()),
            ("p@1", get_p_at_k_with_ties_from_scores, dict(k=1)),
            ("p@3", get_p_at_k_with_ties_from_scores, dict(k=3)),
            ("p@5", get_p_at_k_with_ties_from_scores, dict(k=5)),
            ("p@7", get_p_at_k_with_ties_from_scores, dict(k=7)),
            ("p@10", get_p_at_k_with_ties_from_scores, dict(k=10)),
        ]

        if SPLIT == "dev":
            eval_queries = load_query_data(f"{data_path}/dev_1000_results.json")    
        elif SPLIT == "test":
            eval_queries = load_query_data(f"{data_path}/test_1000_results.json")

            
        print(f"processing {SIMILARITY_NAME} similarity: {split}")
        print(cache_file)
        Sim = syntax_similarity.Similarity()
        print(len(Sim.cache))
        run_nq(
            Sim.similarity,
            SIMILARITY_NAME,
            train_queries,
            eval_queries,
            tuples,
            k_values,
            n_values,
            metrics,
            f"./plots/{dataset}/nq",
            PERCENT,
            SPLIT
        )

if __name__ == "__main__":
    main()
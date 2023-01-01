from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score


def get_kendall_tau_from_scores(predicted_scores, gold_scores, use_lineage=False):
    """

    :param predicted_scores: sorted list of (tuple_id, score) pairs
    :param gold_scores:
    :param use_lineage:
    :return:
    """

    if use_lineage:
        gold_rank = [i + 1 for i in range(len(gold_scores))]

        predicted_tuples = {t[0]: t for t in predicted_scores}
        lineage_scores = []
        for tuple_id, _ in gold_scores:
            if tuple_id in predicted_tuples:
                tuple_score = predicted_tuples[tuple_id]
            else:
                tuple_score = (tuple_id, 0)
            lineage_scores.append(tuple_score)

        lineage_scores.sort(key=lambda x: (-x[1], x[0]))
        lineage_rank_data = {t[0]: i + 1 for i, t in enumerate(lineage_scores)}

        lineage_rank = [lineage_rank_data[t[0]] for t in gold_scores]
        coef, p = kendalltau(gold_rank, lineage_rank)
        return coef



def get_ndcg_from_scores(predicted_scores, gold_scores, use_lineage=False):
    if use_lineage:
        gold_scores_values = [t[1] for t in gold_scores]

        predicted_tuples = {t[0]: t for t in predicted_scores}
        lineage_scores = []
        for tuple_id, _ in gold_scores:
            if tuple_id in predicted_tuples:
                tuple_score = predicted_tuples[tuple_id][1]
            else:
                tuple_score = 0
            lineage_scores.append(tuple_score)

        return ndcg_score([gold_scores_values], [lineage_scores])


def get_p_at_k_from_scores(sorted_predictions, gold_scores, k=10, use_lineage=True):
    gold_tuple_ids = set([t[0] for t in gold_scores[:k]])
    predicted_tuple_ids = set([t[0] for t in sorted_predictions[:k]])
    predicted_in_top = [p for p in predicted_tuple_ids if p in gold_tuple_ids]
    return len(predicted_in_top) / k




def get_p_at_k_with_ties_from_scores(sorted_predictions, gold_scores, k=10, use_lineage=True):
    top_k = gold_scores[:k]
    if len(gold_scores) > k:
        i = k
        while i < len(gold_scores) and gold_scores[i][1] == top_k[-1][1]:
            i += 1
        top_k += gold_scores[k:i+1]

    gold_tuple_ids = set([t[0] for t in top_k])
    predicted_tuple_ids = set([t[0] for t in sorted_predictions[:k]])
    predicted_in_top = [p for p in predicted_tuple_ids if p in gold_tuple_ids]
    return len(predicted_in_top) / k

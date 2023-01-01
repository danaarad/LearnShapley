import json
from time import time
import torch

from models.utils import get_kendall_tau_from_scores, get_ndcg_from_scores
from models.utils import get_p_at_k_with_ties_from_scores


def eval_ranking(model, queries, dataloaders, gold_scores_cache, args, epoch, return_avgs=True):
    query_names = []
    kt_values = []
    ndcg_values = []
    kendalltau_avgs = []
    p_at_1_avgs = []
    p_at_3_avgs = []
    p_at_5_avgs = []

    ndcg_avgs = []
    triplets_count = 0

    k = args.topk

    eval_times = []
    for query in queries:
        kendalltau_query_values = []
        ndcg_query_values = []
        p_at_1_values = []
        p_at_3_values = []
        p_at_5_values = []

        if not query.results:
            continue
        
        for result_index, result in enumerate(query.results):
            start_time = time()
            dataloader = dataloaders[(query.query_name, result.tuple_id)]
            
            result.facts.sort(key=lambda x: (-x.shapley_value, x.tuple_id))
            gold_scores = gold_scores_cache[(query.query_name, result.tuple_id)][:k]
            fact_ids = [t[0] for t in gold_scores]

            predictions = []
            
            with torch.no_grad():
                for batch_index, (data, attn, labels) in enumerate(dataloader):
                    data, attn, labels = data.to(args.device), attn.to(args.device), labels.to(args.device)
                    if args.model == "transformer":
                        output = model(data, src_key_padding_mask=attn)
                        output = output.to(args.device)[:, 0, :]
                        output = torch.squeeze(output).cpu().tolist()
                    elif args.model == "bert":
                        output = model(data, attention_mask=attn)
                        output = output.logits
                        output = torch.squeeze(output).tolist()
                    elif args.model == "bert_shap":
                        output = model(data, attention_mask=attn)
                        output = torch.squeeze(output).tolist()
                
                    if isinstance(output, list):
                        predictions.extend(output)
                        triplets_count += len(output)
                    else:
                        predictions.append(output)
                        triplets_count += 1
            
            end_time = time()
            eval_time = end_time - start_time
            eval_times.append(eval_time)

            sorted_predictions = list(zip(fact_ids, predictions))
            sorted_predictions.sort(key=lambda x: (-x[1], x[0]))
            sorted_predictions = sorted_predictions[:k]

            kt = get_kendall_tau_from_scores(sorted_predictions, gold_scores, use_lineage=True)
            kendalltau_query_values.append(kt)

            ndcg = get_ndcg_from_scores(sorted_predictions, gold_scores, use_lineage=True)
            ndcg_query_values.append(ndcg)
            
            p_at_1 = get_p_at_k_with_ties_from_scores(sorted_predictions, gold_scores, 1)
            p_at_1_values.append(p_at_1)

            p_at_3 = get_p_at_k_with_ties_from_scores(sorted_predictions, gold_scores, 3)
            p_at_3_values.append(p_at_3)

            p_at_5 = get_p_at_k_with_ties_from_scores(sorted_predictions, gold_scores, 5)
            p_at_5_values.append(p_at_5)

        query_names.append(query.query_name)
        kt_values.append(kendalltau_query_values)
        ndcg_values.append(ndcg_query_values)
                    
        kendalltau_avgs.append(sum(kendalltau_query_values) / len(query.results))
        ndcg_avgs.append(sum(ndcg_query_values) / len(query.results))
        p_at_1_avgs.append(sum(p_at_1_values) / len(query.results))
        p_at_3_avgs.append(sum(p_at_3_values) / len(query.results))
        p_at_5_avgs.append(sum(p_at_5_values) / len(query.results))
        
    print(f"DONE. avg time per query + output tuple {sum(eval_times)/len(eval_times)}, max time for query + output tuple: {max(eval_times)}")
    if return_avgs:
        return kendalltau_avgs, ndcg_avgs, p_at_1_avgs, p_at_3_avgs, p_at_5_avgs, 0
    else:
        return kt_values, ndcg_values

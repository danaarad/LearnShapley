import time
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

from transformers import BertTokenizer
from transformers import AutoModelForSequenceClassification

from models.dataset import create_query_result_tensors, process_attention_masks
from models.eval import  eval_ranking
from models.learnshapley.finetuning import BertShapModel
from structures.query_log import load_query_data, get_tuples_to_values_mapping

import similarity_functions.syntax_based_similarity as syntax_similarity
import similarity_functions.witness_based_similarity as witness_similarity
import similarity_functions.rank_based_similarity as rank_similarity


def eval(args):
    checkpoints_dir = glob(f"{args.eval_checkpoints}/*")
    if args.model == "bert_shap":
        checkpoints = []
        for checkpoint in checkpoints_dir:
            pytorch_bin_file = [fname for fname in glob(f"{checkpoint}/*") if "pytorch_model.bin" in fname]
            checkpoints.extend(pytorch_bin_file)        
    else:
        checkpoints = [fname for fname in checkpoints_dir if "checkpoint" in fname]

    # TODO
    # checkpoints = [
    #     "./models/bert/sim_0_3w_0_7s/5k_shap/checkpoint-128854/pytorch_model.bin",
    #     "./models/bert/sim_r/5k_shap/checkpoint-128854/pytorch_model.bin"
    # ]

    print("processing checkpoints:")
    checkpoint_names = [c.replace(f"{args.eval_checkpoints}/", "") for c in checkpoints]
    for checkpoint in checkpoint_names:
        print(f"\t--{checkpoint}")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("loading tuples")
    tuples = get_tuples_to_values_mapping(f"{args.data}/train_tuples.json")

    if args.eval_split == "dev":
        queries_file = f"{args.data}/dev_{args.max_results_for_eval}_results.json"
        dev_tuples = get_tuples_to_values_mapping(f"{args.data}/dev_tuples.json")
        tuples.update(dev_tuples)
    else:
        queries_file = f"{args.data}/test_{args.max_results_for_eval}_results.json"
        test_tuples = get_tuples_to_values_mapping(f"{args.data}/test_tuples.json")
        tuples.update(test_tuples)
    print(f"loaded {len(tuples)} tuples")

    print("creating eval cache")
    queries = load_query_data(queries_file)
    eval_dataloaders = dict()
    gold_scores_cache = dict()

    for query in tqdm(queries):
        for result_index, result in enumerate(query.results[:args.max_results_for_eval]):
            if len(result.facts) < 2:
                continue
            input_ids, attention_masks, labels = create_query_result_tensors(
                query,
                result_index,
                tuples,
                tokenizer,
                args
            )
            attention_masks = process_attention_masks(attention_masks)
            dataset = TensorDataset(input_ids, attention_masks, labels)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            eval_dataloaders[(query.query_name, result.tuple_id)] = dataloader

            result.facts.sort(key=lambda x: (-x.shapley_value, x.tuple_id))
            gold_scores = [(f.tuple_id, f.shapley_value) for f in result.facts]
            gold_scores_cache[(query.query_name, result.tuple_id)] = gold_scores

    
    for checkpoint_name, checkpoint in zip(checkpoint_names, checkpoints):
        print(f"evaluating checkpoint: {checkpoint}")
        if args.model == "bert_shap":
            model = BertShapModel(args).to(args.device)
            model.load_state_dict(torch.load(checkpoint))
        elif args.model == "bert":
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(args.device)
        # TODO
        
        dev_avg_kendalltaus, dev_avg_ndcgs, dev_p_at_1s, dev_p_at_3s, dev_p_at_5s, avg_eval_time = eval_ranking(model, queries, eval_dataloaders, gold_scores_cache, args, checkpoint_name)
        
        avg_kendalltau = sum(dev_avg_kendalltaus) / len(dev_avg_kendalltaus)
        avg_ndcg = sum(dev_avg_ndcgs) / len(dev_avg_ndcgs)
        avg_p_at_1 = sum(dev_p_at_1s) / len(dev_p_at_1s)
        avg_p_at_3 = sum(dev_p_at_3s) / len(dev_p_at_3s)
        avg_p_at_5 = sum(dev_p_at_5s) / len(dev_p_at_5s)

        print(f"checkpoint_name {checkpoint_name}: ")
        print(f"\t kt={avg_kendalltau}, ndcg={avg_ndcg}")
        print(f"\t p@1={avg_p_at_1}, p@3={avg_p_at_3}, p@5={avg_p_at_5}")
        with open(f"{args.save}/{args.model_name}_{args.eval_split}_log.csv", "ab") as f:
            line_data = [args.eval_split, checkpoint_name, avg_kendalltau, avg_ndcg, avg_p_at_1, avg_p_at_3, avg_p_at_5, avg_eval_time]
            line_data = [str(x) for x in line_data]
            line = f"{','.join(line_data)}\n".encode()
            f.write(line)
    
        # _, ndcg_values = eval_ranking(model, queries, eval_dataloaders, gold_scores_cache, args, checkpoint_idx, return_avgs=False)
        # plot_ndcg_vs_nq_dist(queries, ndcg_values, tuples, args, checkpoint_idx)


def plot_ndcg_vs_nq_dist(eval_queries, ndcg_values, tuples, args, checkpoint_idx):
    train_queries = load_query_data(f"{args.data}/train_{args.max_results_for_train}_results.json")

    similarity_functions = [
    ("syntax", "./similarity_functions/similarity_cache/syntax_similarity_cache.json", syntax_similarity),
    # ("witness", "./similarity_functions/similarity_cache/witness_similarity_cache.json", witness_similarity),
    # ("rank", "./similarity_functions/similarity_cache/rank_similarity_cache.json", rank_similarity)
    ]

    similarities = []

    for query in tqdm(eval_queries):
        if query.results:
            query_similarities = []
            for result_index, result in enumerate(query.results):
                for similarity_name, cache_file, similarity_module in similarity_functions:
                    print(f"processing {similarity_name} similarity")
                    similarity_module.load_cache(cache_file)

                    output_value = tuples[result.tuple_id]

                    nearest_similarity = 0
                    for other_query in train_queries:
                        curr_similarity = similarity_module.similarity(query, other_query)
                        if curr_similarity > nearest_similarity:
                            for result in other_query.results:
                                result_value = tuples[result.tuple_id]
                                if result_value == output_value:   
                                    nearest_similarity = curr_similarity
                    
                    query_similarities.append(nearest_similarity)
        similarities.append(query_similarities)

    plt.figure().clear()
    plt.title(f"NDCG vs. Similarity Score w.r.t Nearest Query ({checkpoint_idx})")
    for x,y in zip(ndcg_values, similarities):
        plt.scatter(x, y)
    
    plt.ylabel(f"{similarity_name} Similarity Score")
    plt.xlabel(f"NDCG Score")
    # plt.xticks(rotation=45)
    # plt.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(f"./plots/ndcg_vs_similarity_{checkpoint_idx}.png", format="png")








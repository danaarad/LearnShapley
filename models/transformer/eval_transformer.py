from glob import glob
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

from transformers import BertTokenizer

from models.transformer.model import TransformerModel
from models.dataset import create_query_result_tensors, process_attention_masks
from models.eval import  eval_ranking
from structures.query_log import load_query_data, get_tuples_to_values_mapping


# checkpoints = [
#   "./models/transformer/3_layers_5k_100p/emsize128_nhid128_nlayers3_nhead8_bs128_lr0.0001_m0.9_dr0.2_100precent_20_state_dict.pt",
# #   "./models/transformer/3_layers_10k/emsize128_nhid128_nlayers3_nhead8_bs64_lr0.0001_schedFalse_m0.9_dr0.2_5_model.pt",
# #   "./models/transformer/3_layers_10k/emsize128_nhid128_nlayers3_nhead8_bs64_lr0.0001_schedFalse_m0.9_dr0.2_6_model.pt",
# #   "./models/transformer/3_layers_10k/emsize128_nhid128_nlayers3_nhead8_bs64_lr0.0001_schedFalse_m0.9_dr0.2_8_model.pt",
# #   "./models/transformer/3_layers_10k/emsize128_nhid128_nlayers3_nhead8_bs64_lr0.0001_schedFalse_m0.9_dr0.2_10_model.pt",
# #   "./models/transformer/3_layers_10k/emsize128_nhid128_nlayers3_nhead8_bs64_lr0.0001_schedFalse_m0.9_dr0.2_13_model.pt",
# #   "./models/transformer/3_layers_10k/emsize128_nhid128_nlayers3_nhead8_bs64_lr0.0001_schedFalse_m0.9_dr0.2_16_model.pt"
# ]

# epochs = [20]


def eval(args):
    checkpoints = [fname for fname in glob(f"{args.eval_checkpoints}/*") if "state_dict.pt" in fname]
    checkpoint_names = [c.replace(f"{args.eval_checkpoints}/", "") for c in checkpoints]       
    print("processing checkpoints:")
    for checkpoint in checkpoint_names:
        print(f"\t--{checkpoint}")
        
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    args.ntokens = len(tokenizer.vocab)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    for epoch, checkpoint in zip(checkpoint_names, checkpoints):
        print(f"evaluating checkpoint: {checkpoint}")
        model = TransformerModel(args).to(args.device)
        model.load_state_dict(torch.load(checkpoint))

        dev_avg_kendalltaus, dev_avg_ndcgs, dev_p_at_1s, dev_p_at_3s, dev_p_at_5s, avg_eval_time = eval_ranking(model, queries, eval_dataloaders, gold_scores_cache, args, epoch)
        
        avg_kendalltau = sum(dev_avg_kendalltaus) / len(dev_avg_kendalltaus)
        avg_ndcg = sum(dev_avg_ndcgs) / len(dev_avg_ndcgs)
        avg_p_at_1 = sum(dev_p_at_1s) / len(dev_p_at_1s)
        avg_p_at_3 = sum(dev_p_at_3s) / len(dev_p_at_3s)
        avg_p_at_5 = sum(dev_p_at_5s) / len(dev_p_at_5s)

        print(f"epoch {epoch}: ")
        print(f"\t kt={avg_kendalltau}, ndcg={avg_ndcg}")
        print(f"\t p@1={avg_p_at_1}, p@3={avg_p_at_3}, p@5={avg_p_at_5}")
        with open(f"{args.save}/{args.model_name}_{args.eval_split}_log.csv", "ab") as f:
            line_data = [args.eval_split, epoch, avg_kendalltau, avg_ndcg, avg_p_at_1, avg_p_at_3, avg_p_at_5, avg_eval_time]
            line_data = [str(x) for x in line_data]
            line = f"{','.join(line_data)}\n".encode()
            f.write(line)
        

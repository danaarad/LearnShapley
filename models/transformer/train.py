import time
import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

from transformers import BertTokenizer

from models.transformer.model import TransformerModel
from models.dataset import create_query_result_tensors, process_attention_masks
from models.eval import eval_ranking
from structures.query_log import load_query_data, get_tuples_to_values_mapping


#
# Code adapted from pytorch examples: Word-level language modeling RNN
# https://github.com/pytorch/examples/blob/151944ecaf9ba2c8288ee550143ae7ffdaa90a80/word_language_model/main.py
#

def get_dataset_tensors(args, split):
    if split == "train":
        max_results = args.max_results_for_train
        percent = f"{args.queries_percent_for_train}_queries_" if args.queries_percent_for_train != 100 else ""
    elif split == "dev":
        max_results = args.max_results_for_eval
        percent = ""
    else:
        raise Exception(split)
    
    inp_filename = f"{args.data}/input_ids_{max_results}_results_{percent}{split}.pt"
    attn_filename = f"{args.data}/attn_{max_results}_results_{percent}{split}.pt"
    labels_filename = f"{args.data}/labels_{max_results}_results_{percent}{split}.pt"

    print("loading tensors from files:")
    for filename in [inp_filename, attn_filename, labels_filename]:
        print(filename)

    input_ids = torch.load(inp_filename)
    attention_masks = torch.load(attn_filename)
    labels = torch.load(labels_filename)

    attention_masks = process_attention_masks(attention_masks)
    return input_ids, attention_masks, labels


def evaluate_loss(args, model, criterion, dataloader):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_index, (data, attn, labels) in enumerate(dataloader):
            data, attn, labels = data.to(args.device), attn.to(args.device), labels.to(args.device)
            output = model(data, src_key_padding_mask=attn)
            output = output.to(args.device)[:, 0, :]
            output = torch.squeeze(output)
            total_loss += criterion(output, labels).item()
    return total_loss / len(dataloader)


def train_epoch(args, model, criterion, optimizer, epoch, dataloader):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch_index, (data, attn, labels) in enumerate(dataloader):
        data, attn, labels = data.to(args.device), attn.to(args.device), labels.to(args.device)
        # data.shape (batch_size, seq_size)
        # labels.shape (batch_size, 1)

        optimizer.zero_grad()
        output = model(data, src_key_padding_mask=attn)
        # output.shape (batch_size, seq_size, 1)

        output = output.to(args.device)[:, 0, :]
        output = torch.squeeze(output)
        # output.shape (batch_size, 1)


        # The output is expected to contain scores for each class.
        # output has to be a 2D Tensor of size (minibatch, C).
        # This criterion expects a class index (0 to C-1) as the target
        # for each value of a 1D tensor of size minibatch
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_index % args.log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / batch_index
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '
                  'loss {:3.8f} '.format(epoch, batch_index, len(dataloader), args.lr,
                                         elapsed * 1000 / args.log_interval, cur_loss))
            start_time = time.time()


def train(args):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    args.ntokens = len(tokenizer.vocab)
    print("loading tuples")
    tuples = get_tuples_to_values_mapping(f"{args.data}/train_tuples.json")
    dev_tuples = get_tuples_to_values_mapping(f"{args.data}/dev_tuples.json")
    tuples.update(dev_tuples)
    print(f"loaded {len(tuples)} tuples")

    input_ids, attention_masks, labels = get_dataset_tensors(args, "train")
    train_data = TensorDataset(input_ids, attention_masks, labels)
    input_ids, attention_masks, labels = get_dataset_tensors(args, "dev")
    dev_data = TensorDataset(input_ids, attention_masks, labels)
    print(f"loaded {len(train_data)} train triplets, {len(dev_data)} dev triplets")

    print("creating dataloaders")
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=True)

    print("creating eval cache")
    dev_queries_file = f"{args.data}/dev_{args.max_results_for_eval}_results.json"
    dev_queries = load_query_data(dev_queries_file)
    dev_eval_dataloaders = dict()
    gold_scores_cache = dict()

    for query in tqdm(dev_queries):
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
            dev_eval_dataloaders[(query.query_name, result.tuple_id)] = dataloader

            result.facts.sort(key=lambda x: (-x.shapley_value, x.tuple_id))
            gold_scores = [(f.tuple_id, f.shapley_value) for f in result.facts]
            gold_scores_cache[(query.query_name, result.tuple_id)] = gold_scores

    ###############################################################################
    # Build the model
    ###############################################################################

    model = TransformerModel(args).to(args.device)
    criterion = nn.MSELoss().to(args.device)

    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr
    )

    ###############################################################################
    # Training code
    ###############################################################################

    print("training")
    try:
        best_dev_ndcg = None
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_epoch(args, model, criterion, optimizer, epoch, train_dataloader)

            train_loss = evaluate_loss(args, model, criterion, train_dataloader)
            dev_loss = evaluate_loss(args, model, criterion, dev_dataloader)
            _, dev_avg_ndcgs, dev_p_at_1s, dev_p_at_3s, dev_p_at_5s, avg_eval_time = eval_ranking(model, dev_queries, dev_eval_dataloaders, gold_scores_cache, args, epoch)
            avg_ndcg = sum(dev_avg_ndcgs) / len(dev_avg_ndcgs)
            avg_p_at_1 = sum(dev_p_at_1s) / len(dev_p_at_1s)
            avg_p_at_3 = sum(dev_p_at_3s) / len(dev_p_at_3s)
            avg_p_at_5 = sum(dev_p_at_5s) / len(dev_p_at_5s)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | dev loss {:3.8f} | avg dev kendall tau {} | avg dev ndcg {}'.format(
                epoch, (time.time() - epoch_start_time), dev_loss, avg_kendalltau, avg_ndcg))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_dev_ndcg or avg_ndcg > best_dev_ndcg:
                with open(f"{args.save}/{args.model_name}_{epoch}_state_dict.pt", 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_dev_ndcg = avg_ndcg

            # save loss and metrics
            with open(f"{args.save}/{args.model_name}_log.csv", "ab") as f:
                line = f"{epoch},{train_loss},{dev_loss},{avg_ndcg},{avg_p_at_1},{avg_p_at_3},{avg_p_at_5}{avg_eval_time}\n".encode()
                f.write(line)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

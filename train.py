import os
import argparse
from transformers import BertTokenizer

from models.transformer.train import train as t_train
from models.transformer.train import *
from models.bert.train import train as b_train
from models.learnshapley.bert_sim import train as b_sim_train
from models.learnshapley.bert_shap import train as b_shap_train

from models.bert.eval_bert import eval as b_eval
from models.learnshapley.eval_bert import eval as ls_eval
from models.transformer.eval_transformer import eval as t_eval

from models.dataset import create_tensors, create_similarity_tensors
from structures.query_log get_tuples_to_values_mapping


def train(args):
    if args.model == "transformer":
        model_name = f"emsize{args.emsize}_nhid{args.nhid}_nlayers{args.nlayers}_nhead{args.nhead}"
        model_name += f"_bs{args.batch_size}_lr{args.lr}_m{args.momentum}_dr{args.dropout}"
        model_name += f"_{args.queries_percent_for_train}precent"
        args.model_name = model_name
        print(f"starting {args.model} with {model_name.replace('_', ' ')}")
        t_train(args)
    elif args.model == "bert":
        args.model_name = f"bert_{args.max_results_for_train}_results_{args.batch_size}_bs"
        b_train(args)
    elif args.model == "bert_sim":
        args.model_name = f"bert_sim_{args.max_results_for_train}_results_{args.batch_size}_bs"
        b_sim_train(args)
    elif args.model == "bert_shap":
        args.model_name = f"bert_shap_{args.max_results_for_train}_results_{args.batch_size}_bs"
        b_shap_train(args)


def eval(args):
    if args.model == "transformer":
        args.model_name = f"{args.nlayers}nlayers_{args.max_results_for_train}_results"
        t_eval(args)
    elif args.model == "bert":
        args.model_name = f"{args.model}_{args.max_results_for_train}_results"
        b_eval(args)
    elif args.model == "bert_shape":
        args.model_name = f"{args.model}_{args.max_results_for_train}_results"
        ls_eval(args)


def create_data(args):
    args.load_tensors = False
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    args.ntokens = len(tokenizer.vocab)
    
    print("loading tuples")
    tuples = get_tuples_to_values_mapping(f"{args.data}/train_tuples.json")
    dev_tuples = get_tuples_to_values_mapping(f"{args.data}/dev_tuples.json")
    test_tuples = get_tuples_to_values_mapping(f"{args.data}/test_tuples.json")
    tuples.update(dev_tuples)
    tuples.update(test_tuples)
    print(f"loaded {len(tuples)} tuples")


    train_queries_file = f"{args.data}/train.json"
    print(f"creating shap tensors from: {train_queries_file}")
    input_ids, attention_masks, labels = create_tensors(
        train_queries_file,
        tuples,
        tokenizer,
        args,
        f"train"
    )
    print(f"creating similarity tensors from: {train_queries_file}")
    _ = create_similarity_tensors(
        train_queries_file,
        tokenizer,
        args,
        "train"
    )

    dev_queries_file = f"{args.data}/dev.json"
    print(f"creating shap tensors from: {dev_queries_file}")
    input_ids, attention_masks, labels = create_tensors(
        dev_queries_file,
        tuples,
        tokenizer,
        args,
        "dev"
    )
    dev_queries_file = f"{args.data}/dev.json"
    print(f"creating similarity tensors from: {dev_queries_file}")
    _ = create_similarity_tensors(
        dev_queries_file,
        tokenizer,
        args,
        "dev"
    )

    test_queries_file = f"{args.data}/test.json"
    print(f"creating shap tensors from: {test_queries_file}")
    input_ids, attention_masks, labels = create_tensors(
        test_queries_file,
        tuples,
        tokenizer,
        args,
        "test"
    )


def main():
    parser = argparse.ArgumentParser(description='PyTorch LearnShapley Implementation')
    parser.add_argument('--data', type=str, default='./data/',
                        help='location of the data corpus')
    parser.add_argument('--save', type=str, default='./',
                        help='path to save the final model')
    parser.add_argument('--action', type=str, default="train",
                        help='train, eval, create data')
    parser.add_argument('--model', type=str, default="transformer",
                        help='model to train (transformer or bert)')
    parser.add_argument('--topk', type=int, default=10,
                        help='top k values to rank')

    parser.add_argument('--queries_percent_for_train', type=int, default=100,
                        help='percent of train queries to train on')
    parser.add_argument('--max_results_for_eval', type=int, default=1000,
                        help='max results to use for eval per query')
    parser.add_argument('--max_results_for_train', type=int, default=5000,
                        help='max results to use for eval per query')
    parser.add_argument('--load_tensors', type=bool, default=True,
                        help='load tensors from files')
    parser.add_argument('--sim_checkpoint', type=str, help='checkpoint to load')
    parser.add_argument('--eval_checkpoints', type=str, help='path to checkpoints dir to load')
    parser.add_argument('--eval_split', type=str, default="dev",
                        help='split of data to eval (dev or test)')

    parser.add_argument('--use_sim_s', action='store_true')
    parser.add_argument('--sim_s_weight', type=float, default=1.0)
    parser.add_argument('--use_sim_r', action='store_true')
    parser.add_argument('--sim_r_weight', type=float, default=1.0)
    parser.add_argument('--use_sim_w', action='store_true')
    parser.add_argument('--sim_w_weight', type=float, default=1.0)

    parser.add_argument('--emsize', type=int, default=128,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=128,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=8,
                        help='the number of heads in the encoder/decoder of the transformer model')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='initial momentum')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--grad_accumulation', type=int, default=1,
                        help='Number of updates steps to accumulate the gradients for, before performing a backward/update pass.')


    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='mex sequence length size')

    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='report interval')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')


    args = parser.parse_args()
    print("torch.cuda.is_available(): ", torch.cuda.is_available())
    torch.cuda.empty_cache() 
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.device)
    print(args.data)

    if args.action == "train":
        train(args)
    elif args.action == "eval":
        eval(args)
    elif args.action == "create_data":
        create_data(args)


if __name__ == "__main__":
    main()
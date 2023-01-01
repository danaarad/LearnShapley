import torch
from structures.query_log import load_query_data
from similarity_functions import syntax_based_similarity as sim_s
from similarity_functions import witness_based_similarity as sim_w
from similarity_functions import rank_based_similarity as sim_r

tensors_cache = {}


def process_attention_masks(raw_attention_masks):
    raw_attention_masks = raw_attention_masks == 0
    attention_masks = torch.zeros_like(raw_attention_masks).to(torch.float)
    attention_masks[raw_attention_masks] = float("-inf")
    return attention_masks


def create_tensors(query_log_file, tuples, tokenizer, args, split):
    print(args.queries_percent_for_train)
    input_ids = []
    attention_masks = []
    shapley_values = []

    if split == "train":
        max_results = args.max_results_for_train
        percent = f"{args.queries_percent_for_train}_" if args.queries_percent_for_train != 100 else ""
    elif split == "dev" or split == "test":
        max_results = args.max_results_for_eval
        percent = ""
    else:
        raise Exception(split)

    sep = tokenizer.sep_token

    count = 0
    query_log = load_query_data(query_log_file)
    print(f"{args.queries_percent_for_train} percent: {len(query_log)} queries")
    for query in query_log:
        for result in query.results[:max_results]:
            result_tuple = str(tuples[result.tuple_id])
            for fact in result.facts:
                cache_key = (query.query_name, result.tuple_id, fact.tuple_id)
                if cache_key in tensors_cache:
                    encoded_dict = tensors_cache[cache_key]
                else:
                    fact_tuple = str(tuples[fact.tuple_id])
                    triplet = f"{query.sql} {sep} {result_tuple} {sep} {fact_tuple}"
                    encoded_dict = tokenizer.encode_plus(
                        triplet,
                        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                        max_length=args.max_seq_len,  # Pad & truncate all sentences.
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,  # Construct attn. masks.
                        return_tensors='pt',  # Return pytorch tensors.
                    )
                    tensors_cache[cache_key] = encoded_dict

                # Add the encoded sentence to the list.
                input_ids.append(encoded_dict['input_ids'])

                # And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(encoded_dict['attention_mask'])

                shapley_values.append(fact.shapley_value * 1000)

                count += 1
                if count % 10000 == 0:
                    print(f"processed {count} {split} samples")

    input_ids = torch.cat(input_ids).to(torch.int64)
    attention_masks = torch.cat(attention_masks).to(torch.float)
    labels = torch.tensor(shapley_values).to(torch.float)

    torch.save(input_ids, f"{args.data}/input_ids_{max_results}_results_{percent}{split}.pt")
    torch.save(attention_masks, f"{args.data}/attn_{max_results}_results_{percent}{split}.pt")
    torch.save(labels, f"{args.data}/labels_{max_results}_results_{percent}{split}.pt")
    return input_ids, attention_masks, labels
    

def create_similarity_tensors(query_log_file, tokenizer, args, split):
    input_ids = []
    attention_masks = []
    sim_s_values = []
    sim_r_values = []
    sim_w_values = []

    Sims = sim_s.Similarity(f"./similarity_functions/similarity_cache/{args.dataset}_syntax_similarity_cache.json")
    Simr = sim_r.Similarity("./similarity_functions/similarity_cache/{args.dataset}_rank_similarity_cache.json")
    Simw = sim_w.Similarity("./similarity_functions/similarity_cache/{args.dataset}_witness_similarity_cache.json")

    if split == "train":
        percent = f"{args.queries_percent_for_train}_" if args.queries_percent_for_train != 100 else ""
    elif split == "dev":
        percent = ""
    else:
        raise Exception(split)

    sep = tokenizer.sep_token

    count = 0
    query_log = load_query_data(query_log_file)
    print(f"{args.queries_percent_for_train} percent: {len(query_log)} queries")
    for query1 in query_log:
        query1.results = []
        for query2 in query_log:
            query2.results = []
            if query1.query_name == query2.query_name:
                continue

            queries = f"{query1.sql} {sep} {query2.sql}"
            encoded_dict = tokenizer.encode_plus(
                queries,
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=args.max_seq_len,  # Pad & truncate all sentences.
                padding='max_length',
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            similarity_cache_key = tuple(sorted([query1.query_name, query2.query_name]))

            sim_s_value = Sims.cache.get(similarity_cache_key, -1)
            sim_r_value = Simr.cache.get(similarity_cache_key, -1)
            sim_w_value = Simw.cache.get(similarity_cache_key, -1)

            if sim_w_value is None or sim_r_value is None or sim_s_value is None:
                continue

            sim_r_values.append(sim_r_value)
            sim_s_values.append(sim_s_value)
            sim_w_values.append(sim_w_value)
            
            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

            count += 1
            if count % 1000 == 0:
                print(f"processed {count} {split} samples")

    input_ids = torch.cat(input_ids).to(torch.int64)
    attention_masks = torch.cat(attention_masks).to(torch.float)
    sim_s_values = torch.tensor(sim_s_values).to(torch.float)
    sim_r_values = torch.tensor(sim_r_values).to(torch.float)
    sim_w_values = torch.tensor(sim_w_values).to(torch.float)

    torch.save(input_ids, f"{args.data}/sim_input_ids_{percent}{split}.pt")
    torch.save(attention_masks, f"{args.data}/sim_attn_{percent}{split}.pt")
    torch.save(sim_s_values, f"{args.data}/sim_s_{percent}{split}.pt")
    torch.save(sim_r_values, f"{args.data}/sim_r_{percent}{split}.pt")
    torch.save(sim_w_values, f"{args.data}/sim_w_{percent}{split}.pt")
    
    return input_ids, attention_masks, sim_s_values, sim_r_values, sim_w_values


def create_query_result_tensors(query, result_index, tuples, tokenizer, args):
    input_ids = []
    attention_masks = []
    shapley_values = []
    result = query.results[result_index]
    result_tuple = str(tuples[result.tuple_id])
    sep = tokenizer.sep_token

    result.facts.sort(key=lambda x: (-x.shapley_value, x.tuple_id))

    for fact in result.facts:
        fact_tuple = str(tuples[fact.tuple_id])
        triplet = f"{query.sql} {sep} {result_tuple} {sep} {fact_tuple}"
        encoded_dict = tokenizer.encode_plus(
            triplet,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_seq_len,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        shapley_values.append(fact.shapley_value * 1000)

    input_ids = torch.cat(input_ids).to(torch.int64)
    attention_masks = torch.cat(attention_masks).to(torch.float)
    labels = torch.tensor(shapley_values).to(torch.float)

    return input_ids, attention_masks, labels

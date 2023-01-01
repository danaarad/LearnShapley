import torch
from sturctures.query_log import load_query_data

tensors_cache = {}

def create_tensors(query_log_file, tuples, tokenizer, args, split):
    input_ids = []
    attention_masks = []
    shapley_values = []

    if split == "train":
        max_results = args.max_results_for_train
    elif split == "dev":
        max_results = args.max_results_for_eval
    else:
        raise Exception(split)

    sep = tokenizer.sep_token

    count = 0
    query_log = load_query_data(query_log_file)
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

    torch.save(input_ids, f"./data/input_ids_{max_results}_results_{split}.pt")
    torch.save(attention_masks, f"./data/attn_{max_results}_results_{split}.pt")
    torch.save(labels, f"./data/labels_{max_results}_results_{split}.pt")
    return input_ids, attention_masks, labels


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

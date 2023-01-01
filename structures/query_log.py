import json
from structures.query import Query


def load_query_data(file_name, load_results=True):
    with open(file_name, "rb") as f:
        raw_data = json.load(f)

    data = []
    for main_query in raw_data:
        for raw_query in main_query["projected_queries"]:
            query = Query(**raw_query, load_results=load_results)
            data.append(query)
    return data


def get_tuples_to_values_mapping(file_name):
    with open(file_name, "rb") as f:
        return json.load(f)


def map_tuple_id_to_index(query_log):
    tuple_id_to_ind = dict()
    ind_to_tuple_id = dict()

    index = 0
    for query in query_log:
        for result in query.results:
            if not result.index:
                if result.tuple_id in tuple_id_to_ind:
                    result.index = tuple_id_to_ind[result.tuple_id]
                else:
                    result.index = index
                    tuple_id_to_ind[result.tuple_id] = index
                    ind_to_tuple_id[str(index)] = result.tuple_id
                    index += 1

            for fact in result.facts:
                if not fact.index:
                    if fact.tuple_id in tuple_id_to_ind:
                        fact.index = tuple_id_to_ind[fact.tuple_id]
                    else:
                        fact.index = index
                        tuple_id_to_ind[result.tuple_id] = index
                        ind_to_tuple_id[str(index)] = result.tuple_id
                        index += 1

    return tuple_id_to_ind, ind_to_tuple_id

import os
import sys
from logzero import logger
from tests.common import seed_everything, run_test_functions, get_loader_for_selected_validation_examples
import transformers
import torch

DEADLINE = '2024-12-19'
FILES = ["hw4/results_test.txt"]


def test_database(**kwargs):

    def query_lists(query):
        return {k: v if isinstance(v, list) else [v] for k, v in query.items()}

    def query_strings(query):
        return {k: v[0] if isinstance(v, list) else v for k, v in query.items()}

    from diallama.database import MultiWOZDatabase
    database = MultiWOZDatabase()
    output = {}
    output["time_strs"] = [database.time_str_to_minutes(time_str) for time_str in kwargs["time_strs"]]

    # try out which version of the queries is implemented: lists of strings or single strings
    # TODO check whether it gives less than all results (110)
    # TODO try prepending domain name to the slot names
    working_query_func = None
    for query_func in [query_lists, query_strings]:
        try:
            res = database.query("restaurant", query_func({"food": ["chinese"], "pricerange": ["cheap"]}))
            working_query_func = query_func
            if res:  # take the first one that gives any results
                logger.debug(f'Query style {str(query_func)} giving non-empty results')
                break
            logger.debug(f'Query style {str(query_func)} giving empty results')
        except Exception as e:
            logger.debug(f'Query style {str(query_func)} not working: {e}')
            pass

    if not working_query_func:
        logger.error("No working query style found, both options crash.")
        return output

    output["lookups"] = []
    for lookup in kwargs["lookups"]:
        r = database.query(lookup["domain"], working_query_func(lookup["query"]))
        output["lookups"].append(len(r))

    return output


def test_loader(**kwargs):

    loader, dataset = get_loader_for_selected_validation_examples(selected_items=kwargs["selected_items"], context_len=2)
    output = {}

    # TODO insert, don't expect all domains :-\
    output["utterance"] = [inst["utterance"] for inst in dataset.data]
    output["belief_state"] = [inst["belief_state"] for inst in dataset.data]
    output["database_results"] = [inst["database_results"] for inst in dataset.data]
    collated = [inst for inst in loader][0]  # only one batch
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2", padding_side="left")
    # ids & toks can be Tensors or numpy arrays, will work OK
    # TODO check if the type is BatchEncoding, then behave differently (but still fail somehow!!)
    output["belief_state_ids"] = collated["belief_state"] if isinstance(collated["belief_state"], list) else collated["belief_state"].tolist()
    output["belief_state_toks"] = [tokenizer.convert_ids_to_tokens(toks) for toks in collated["belief_state"]]
    output["database_results_ids"] = collated["database_results"] if isinstance(collated["database_results"], list) else collated["database_results"].tolist()
    output["database_results_toks"] = [tokenizer.convert_ids_to_tokens(toks) for toks in collated["database_results"]]

    return output


def check(files):
    default_seed = 42
    seed_everything(default_seed)
    errors = 0

    # check required output files
    for pattern, matches in files.items():
        if pattern.startswith('hw4/results'):
            if os.path.getsize(matches[0]) < 500:
                logger.warning(f'File {pattern} is too small (<500 bytes).')
                errors += 1

    # run code tests (everything starting with test_*)
    errors += run_test_functions(sys.modules[__name__])

    return errors

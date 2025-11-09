import os
import sys
from logzero import logger
from tests.common import seed_everything, run_test_functions

DEADLINE = '2024-11-11'
FILES = ["hw2/results_test.txt"]


def test_known01(**kwargs):

    from diallama.mw_loader import Dataset
    output = {}
    dataset = Dataset("validation")
    output["val"] = len(dataset)

    return output


def test_known02(batch_size=5, shuffle=True, max_batches=1, **kwargs):

    def get_data_types(obj):
        """Check nested data types; ignores null dict entries."""
        if isinstance(obj, list):
            if all(not isinstance(i, (dict, list)) for i in obj):
                if all(type(i) == type(obj[0]) for i in obj):
                    return 'list-' + str(type(obj[0]))
                return 'list-MIXED'
            return [get_data_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: get_data_types(value) for key, value in obj.items() if obj[key] is not None}
        else:
            return str(type(obj))

    from diallama.mw_loader import Dataset, DataLoader

    batches = []
    dataset = Dataset("validation", context_len=None)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate=False)
    for idx, batch in enumerate(data_loader):
        if idx >= max_batches:
            break
        collate_batch = data_loader.collate_fn(batch)
        # check basically just the type info from the loaded batches
        batch_data = {
            "idx": idx,
            "batch": get_data_types(batch),
            "collate_batch": get_data_types(collate_batch),
        }
        batches.append(batch_data)
    output = {"batches": batches}
    return output


def check(files):
    default_seed = 42
    seed_everything(default_seed)
    errors = 0

    # check required output files
    for pattern, matches in files.items():
        if pattern.startswith('hw2/results'):
            if os.path.getsize(matches[0]) < 500:
                logger.warning(f'File {pattern} is too small (<500 bytes).')
                errors += 1

    # run code tests (everything starting with test_*)
    errors += run_test_functions(sys.modules[__name__])

    return errors

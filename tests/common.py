import numpy as np
import random
import torch
import transformers
import subprocess
import datetime
from logzero import logger
import datasets
import copy
import logzero
import logging
import os
import re
import inspect
import json
import importlib
from diff_match_patch import diff_match_patch
from termcolor import colored

dmp = diff_match_patch()


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def croak(message):
    """Print error message and fail."""
    logger.error(message)
    raise Exception(message)


def git_pull(path):
    """Runs git pull, crashes on fail."""
    logger.info('Running git pull...')
    output = subprocess.run("git pull", cwd=path, shell=True, check=True, capture_output=True)
    if output.returncode != 0:
        logger.error(f'Git Error:\n---------\n{output.stderr.decode("utf-8")}')
        croak('Could not pull %s' % path)


def git_checkout_branch(hw_id, path, branch_name=None):
    """Checks out the desired git branch, crashes on fail."""
    if branch_name is None:
        branch_name = hw_id
    logger.info(f'Checking out {hw_id}...')
    output = subprocess.run(f"git checkout {branch_name} --", cwd=path, shell=True, capture_output=True)
    if output.returncode != 0:
        logger.error(f'Git Error:\n---------\n{output.stderr.decode("utf-8")}')
        croak('Could not checkout branch %s for %s' % (hw_id, path))


def git_check_date(path, deadline):
    """Checks the date of the last commit in git (assuming the correct branch is already chosen), logs any problems found.
    Returns 1 on errors found, 0 otherwise."""
    logger.info('Checking last commit date...')
    output = subprocess.run("git show -s --format=%ct HEAD", cwd=path, shell=True, capture_output=True)
    if output.returncode != 0:
        logger.error(f'Git Error:\n---------\n{output.stderr.decode("utf-8")}')
        croak('Could not get last commit date for %s' % path)

    commit_date = datetime.datetime.fromtimestamp(int(output.stdout.strip()))
    deadline_date = datetime.datetime.strptime(deadline + ' 23:59:59', '%Y-%m-%d %H:%M:%S')

    if (deadline_date - commit_date).days < 0:
        if (deadline_date - commit_date).days < - 14:
            logger.error('Last commit date (%s) is > 14 days past the deadline (%s)' % (commit_date.strftime('%Y-%m-%d %H:%M'), deadline))
        else:
            logger.warn('Last commit date (%s) is past the deadline (%s)' % (commit_date.strftime('%Y-%m-%d %H:%M'), deadline))
        return 1
    logger.info('Last commit date (%s) is before the deadline (%s)' % (commit_date.strftime('%Y-%m-%d %H:%M'), deadline))
    return 0


def check_file_presence(path, files):
    """Checks for the presence of given files under a path prefix. Accepts a list of singular paths or tuples
    (regex pattern, expected number of matches). Logs any problems found. Returns number of errors found."""
    logger.info('Checking for files...')

    errors = 0
    file_matches = {}
    for fname in files:
        # number of files to match a given pattern
        if isinstance(fname, tuple):
            fname, fnum = fname
            dirname, fpattern = os.path.split(fname)
            dirname = os.path.join(path, dirname)
            if not os.path.isdir(dirname):
                logger.warn("Found 0 files matching pattern `%s', expected %d" % (fname, fnum))
                errors += 1
                continue
            matches = [os.path.join(dirname, f) for f in os.listdir(dirname)
                       if os.path.isfile(os.path.join(dirname, f)) and re.match(fpattern, f)]
            if len(matches) != fnum:
                logger.warn("Found %d files matching pattern `%s', expected %d" % (len(matches), fname, fnum))
                errors += 1
            if matches:
                file_matches[fname] = matches
        else:
            if not os.path.isfile(os.path.join(path, fname)):
                logger.warn("Did not find file %s" % fname)
                errors += 1
            else:
                file_matches[fname] = [os.path.join(path, fname)]

    return errors, file_matches


def find_component(config, component_pattern):
    """Find component matching the given regex in the config."""
    return any([re.match(component_pattern,
                         next(iter(c.keys())) if isinstance(c, dict) else c)  # either dict with parameters, or plain string
                for c in config['components']])


def colored_diffs(str1, str2):
    diffs = dmp.diff_main(str1, str2)
    dmp.diff_cleanupSemantic(diffs)
    color_str1 = ''.join([colored(d[1], 'red') if d[0] == -1 else d[1] for d in diffs if d[0] <= 0])
    color_str2 = ''.join([colored(d[1], 'cyan') if d[0] == 1 else d[1] for d in diffs if d[0] >= 0])
    return color_str1, color_str2


def run_test_functions(module):

    errors = 0
    test_functions = dict([m for m in inspect.getmembers(module, inspect.isfunction) if m[0].startswith("test_")])
    base_path = module.__file__[:-3]

    for i, test_id in enumerate(test_functions):
        with open(os.path.join(base_path, test_id + ".json"), "r") as f:
            test_data = json.load(f)

        description = test_data.get("description", "")
        logger.info(f"Running test {i}: {description}")

        try:
            output = test_functions[test_id](**test_data["input"])
            expected_output = test_data["expected_output"]
            if output != expected_output:
                diff = ''
                for key in sorted(set(output.keys()) | set(expected_output.keys())):
                    if output.get(key) == expected_output.get(key):
                        continue
                    if key not in output:
                        diff += f'** Missing: {key}\n'
                    elif key not in expected_output:
                        diff += f'** Superfluous: {key}\n'
                    else:
                        exp, rec = colored_diffs(json.dumps(expected_output[key], ensure_ascii=False), json.dumps(output[key], ensure_ascii=False))
                        diff += f'** Expected {key}:\n{exp}\n'
                        diff += f'** Received {key}:\n{rec}\n'
                logger.warning(f"Test {i} failed:\n{diff}\n")
                errors += 1
        except Exception as e:
            if importlib.util.find_spec('pudb'):  # run pudb if it's available
                import pudb; pudb.pm()
            logger.error(f"Test {i} failed to run {e}")
            import traceback
            traceback.print_exc()
            errors += 1

    return errors


def get_loader_for_selected_validation_examples(selected_items, context_len):
    """Get a loader + dataset for a few examples from the 1st dialogue (selected_items = list of ids) in the validation set."""
    from diallama.mw_loader import Dataset, DataLoader

    # load fake empty data
    logzero.loglevel(logging.WARNING)  # suppress training info messages to save console
    dataset = Dataset("validation", context_len=context_len, cache_dir=os.path.join(os.path.dirname(__file__)))
    # process just the few initial dialogues (so it doesn't take ages)
    orig_data = datasets.load_dataset(path='multi_woz_v22', split="validation", streaming=True)
    max_item = max(selected_items)
    for dialogue in orig_data:
        dataset.data.extend(dataset.parse_dialogue_into_examples(dialogue, context_len=context_len))
        if len(dataset.data) >= max_item:
            break
    logzero.loglevel(logging.DEBUG)
    dataset.data = [dataset.data[i] for i in selected_items]  # choose just a few examples
    loader = DataLoader(dataset, batch_size=len(dataset.data), shuffle=False, collate=True)  # make sure we only have 1 batch

    return loader, dataset


def get_basic_trainer_on_data(data, num_epochs):
    """Initialize a trainer + distilgpt2 model + tokenizer on the given data."""

    from diallama.trainer import Trainer
    from diallama.mw_loader import SPECIAL_TOKENS

    class FakeDataLoader:

        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer

        def __iter__(self):
            for batch in self.data:
                yield copy.copy(batch)

        def __len__(self):
            return len(self.data)

    # NB: these initializations are for testing only, don't use them for your real model!
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("distilgpt2", padding_side="left")
    model = transformers.GPT2LMHeadModel.from_pretrained('distilgpt2')
    spec_tokens = list(SPECIAL_TOKENS.values()) if isinstance(SPECIAL_TOKENS, dict) else SPECIAL_TOKENS  # avoid a case where someone changes the list to a dict
    tokenizer.add_special_tokens({"additional_special_tokens": spec_tokens})  # add special tokens
    model.resize_token_embeddings(new_num_tokens=len(tokenizer))  # make space for the special tokens
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = transformers.get_constant_schedule(optimizer)
    loader = FakeDataLoader(data, tokenizer)
    trainer = Trainer(model, loader, loader, num_epochs, optimizer, scheduler)

    return trainer, model, tokenizer

import os
import sys
from logzero import logger
from tests.common import seed_everything, run_test_functions, get_basic_trainer_on_data, get_loader_for_selected_validation_examples
import torch
import transformers
import copy
import logzero
import logging
import inspect


DEADLINE = '2025-01-12'
FILES = ["hw5/multiwoz_outputs.txt", "hw5/multiwoz_scores.txt"]
MASK_TYPES = ["attention_mask", "context_mask", "belief_mask", "database_mask", "utterance_mask"]


def test_collate(**kwargs):

    output = {}
    loader, _ = get_loader_for_selected_validation_examples(kwargs["selected_items"], context_len=2)

    collated = [inst for inst in loader][0]  # only one batch
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2", padding_side="left")

    # nice string rep of the whole tokenized stuff with all the mask identifiers
    masks = {m: collated[m].numpy().tolist() for m in MASK_TYPES}
    input_ids = collated["input_ids"].numpy().tolist()
    input_toks = [tokenizer.convert_ids_to_tokens(toks) for toks in input_ids]

    batch_str = []
    for ex_cnt in range(len(input_ids)):
        ex_str = []
        for tok_cnt in range(len(input_ids[ex_cnt])):
            tok_str = str(int(input_ids[ex_cnt][tok_cnt]))
            tok_str += ''.join([m[0].upper() for m in MASK_TYPES if masks[m][ex_cnt][tok_cnt]])
            tok_str += ' ' + str(input_toks[ex_cnt][tok_cnt])
            ex_str.append(tok_str)
        batch_str.append(ex_str)

    output["batch_str"] = batch_str
    return output


def test_training(**kwargs):

    from diallama.trainer import GenerationWrapper
    from diallama.mw_loader import SPECIAL_TOKENS
    from diallama.database import MultiWOZDatabase

    def check_special_tokens(data):
        """Given a list/string data, return the special tokens at its ends, or ''s"""
        if isinstance(data, list):  # assume 1-element contexts
            data = data[0] if data else ''
        data = data.strip()
        start, end = '', ''
        for spec_tok in SPECIAL_TOKENS:
            if data.startswith(spec_tok):
                start = spec_tok
            if data.endswith(spec_tok):
                end = spec_tok
        return start, end

    loader, dataset = get_loader_for_selected_validation_examples(list(range(len(kwargs['data']))), context_len=2)
    dataset.data = [copy.copy(dataset.data[0]) for _ in kwargs['data']]  # assume always 1st example in dialogue
    # check any special tokens added to the textual parts of the data
    spec_tok_map = {key: check_special_tokens(dataset.data[0][key]) for key in ['context', 'utterance', 'delex_utterance']}
    # overwrite the actual data, keep any potential additional keys produced by the loader (e.g. external user/system indication)
    for trg, src in zip(dataset.data, kwargs['data']):
        for k in src.keys():
            val = src[k][0] if isinstance(src[k], list) else src[k]  # handle lists gracefully
            if k in spec_tok_map:  # re-add special tokens
                val = spec_tok_map[k][0] + val + spec_tok_map[k][1]
            if isinstance(trg[k], list):  # handle lists gracefully
                val = [val]
            trg[k] = val
    # force collation, duplicate but stay in 1 epoch (to save console)
    data = [inst for inst in loader] * kwargs['epochs']

    trainer, model, tokenizer = get_basic_trainer_on_data(data, num_epochs=1)
    # overfit the model on this data
    logzero.logger.info('Running training, please wait...')
    logzero.loglevel(logging.WARNING)  # suppress training info messages to save console
    trainer.train()
    logzero.loglevel(logging.DEBUG)

    # testing that the model is really overfit
    model.eval()
    with torch.no_grad():
        logits = model(**{'attention_mask': data[0]['attention_mask'], 'input_ids': data[0]['input_ids']})[0]
    predicts = torch.argmax(logits, dim=2).numpy()
    output = {}
    # masking out belief & utterance, moving the mask backward 1 token (to get the "next" tokens)
    output["predicted_tokens"] = tokenizer.convert_ids_to_tokens(
        [pred_tok for pred_tok, mask in zip(predicts[0],
                                            torch.roll(data[0]['utterance_mask'][0] | data[0]['belief_mask'][0], -1))
         if mask]
    )

    # trying generation wrapper
    db = MultiWOZDatabase()
    args = [model, tokenizer, kwargs['max_length']]
    sig = inspect.signature(GenerationWrapper.__init__)
    dbpar_idx = [i for i, (k, v) in enumerate(sig.parameters.items())
                 if (k.lower().startswith('db') or k.lower().startswith('database') or 'MultiWOZDatabase' in str(v.annotation))]
    if dbpar_idx:  # 1 db parameter inserted into the signature, if applicable
        assert len(dbpar_idx) == 1
        args.insert(dbpar_idx[0] - 1, db)  # discount `self`
    gen = GenerationWrapper(*args)
    output['generated_text'] = gen.generate_single(kwargs['prompt'])

    return output


def check(files):
    default_seed = 42
    seed_everything(default_seed)
    errors = 0

    # check required output files
    for pattern, matches in files.items():
        if pattern.startswith('hw5/multiwoz_outputs'):
            if os.path.getsize(matches[0]) < 500:
                logger.warning(f'File {pattern} is too small (<500 bytes).')
                errors += 1
        elif pattern.startswith('hw5/multiwoz_scores'):
            if os.path.getsize(matches[0]) < 50:
                logger.warning(f'File {pattern} is too small (<50 bytes).')
                errors += 1

    # run code tests (everything starting with test_*)
    errors += run_test_functions(sys.modules[__name__])

    return errors

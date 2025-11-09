import os
import sys
from logzero import logger
from tests.common import seed_everything, run_test_functions, get_basic_trainer_on_data, get_loader_for_selected_validation_examples
import torch
import transformers
import copy
import logzero
import logging

DEADLINE = '2024-12-12'
FILES = ["hw3/multiwoz_outputs.txt", "hw3/multiwoz_scores.txt"]


def test_collate(**kwargs):

    output = {}
    loader, _ = get_loader_for_selected_validation_examples(kwargs["selected_items"], context_len=2)

    collated = [inst for inst in loader][0]  # only one batch
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2", padding_side="left")
    output["input_ids"] = collated["input_ids"].numpy().tolist()
    output["input_toks"] = [tokenizer.convert_ids_to_tokens(toks) for toks in collated["input_ids"].numpy().tolist()]
    output["attention_mask"] = collated["attention_mask"].numpy().tolist()
    output["context_mask"] = collated["context_mask"].numpy().tolist()
    output["utterance_mask"] = collated["utterance_mask"].numpy().tolist()

    return output


class FakeDataLoader:

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __iter__(self):
        for batch in self.data:
            yield copy.copy(batch)

    def __len__(self):
        return len(self.data)


def test_training(**kwargs):

    from diallama.trainer import GenerationWrapper

    # faking some data
    data = [
        {'attention_mask': torch.tensor(kwargs['attention_mask']).bool(),
         'input_ids': torch.tensor(kwargs['input_ids']),
         'context_mask': torch.tensor(kwargs['context_mask']).bool(),
         'utterance_mask': torch.tensor(kwargs['utterance_mask']).bool(),
         'labels': torch.tensor(kwargs['input_ids']), }
    ]

    trainer, model, tokenizer = get_basic_trainer_on_data(data, num_epochs=10)
    # overfit the model on this data
    logzero.loglevel(logging.WARNING)  # suppress training info messages to save console
    trainer.train()
    logzero.loglevel(logging.DEBUG)
    # testing that the model is really overfit
    model.eval()
    with torch.no_grad():
        logits = model(**{'attention_mask': data[0]['attention_mask'], 'input_ids': data[0]['input_ids']})[0]
    predicts = torch.argmax(logits, dim=2).numpy()
    output = {}
    output["predicted_tokens"] = tokenizer.convert_ids_to_tokens(predicts[0])[kwargs['cut_start']:kwargs['cut_end']]

    # trying generation wrapper (intentionally cutting the text short to test whether max_length is observed)
    gen = GenerationWrapper(model, tokenizer, kwargs['max_length'])
    output['generated_text'] = gen.generate_single(kwargs['prompt'])

    return output


def check(files):
    default_seed = 42
    seed_everything(default_seed)
    errors = 0

    # check required output files
    for pattern, matches in files.items():
        if pattern.startswith('hw3/multiwoz_outputs'):
            if os.path.getsize(matches[0]) < 500:
                logger.warning(f'File {pattern} is too small (<500 bytes).')
                errors += 1
        elif pattern.startswith('hw3/multiwoz_scores'):
            if os.path.getsize(matches[0]) < 50:
                logger.warning(f'File {pattern} is too small (<50 bytes).')
                errors += 1

    # run code tests (everything starting with test_*)
    errors += run_test_functions(sys.modules[__name__])

    return errors

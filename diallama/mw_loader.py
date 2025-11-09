import pickle
import os

import datasets
import torch

from logzero import logger
from diallama.database import MultiWOZDatabase


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class, inherits from torch.utils.data.Dataset.
    Load the MultiWOZ dataset using Huggingface datasets.
    Able to shorten the context length by setting context_len, able to filter by a given domain.
    """

    def __init__(self, split, domain=None, context_len=None, cache_dir="./"):
        """
        Initializes the dataset. If the dataset has already been preprocessed, loads it from the cache.

        :param split: str, dataset split -- one of 'train', 'validation', 'test'
        :param domain: str, domain to filter by, e.g. 'restaurant', 'hotel' (None for no filtering)
        :param context_len: int, maximum number of dialogue turns to include in the context
        :param cache_dir: str, directory to store the preprocessed data ("{split}_{domain}_preprocessed_data.pickle")
        """
        self.split = split
        self.fields = {}
        # Create cache dir if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.f_name = os.path.join(cache_dir, f"{split}_{domain + '_' if domain else ''}preprocessed_data.pickle")
        self.database = MultiWOZDatabase()
        # If the dataset has already been preprocessed, load it from the cache
        if os.path.isfile(self.f_name):
            data = pickle.load(open(self.f_name, 'rb'))
            logger.warn(f"Loaded {len(data)} examples from cached file {self.f_name}. Delete the file to reprocess the data.")
        else: 
            # (HW2): trust remote code set to true
            dataset = datasets.load_dataset(path='multi_woz_v22', split=split, streaming=True, trust_remote_code=True)
            data = []
            for idx, dialogue in enumerate(dataset):
                if idx % 500 == 0:
                    logger.info(f"Processing dialogue {idx + 1}")
                # if limiting to a given domain, skip dialogues that do not belong there
                if domain and (len(dialogue['services']) != 1 or dialogue['services'][0] != domain):
                    continue
                # parse the dialogue
                data.extend(self.parse_dialogue_into_examples(dialogue, context_len=context_len))
            # save a pickle
            self.save_data(data)
        self.data = data

    def save_data(self, data):
        assert not os.path.exists(self.f_name), f"{self.f_name} already exists."
        logger.info(f"Saving {len(data)} examples to {self.f_name}.")
        with open(self.f_name, 'wb+') as f:
            pickle.dump(data, f)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def delexicalize(self, utterance, dialogue_act):
        """
        Replaces slot values in utterances with placeholders to achieve their delexicalized versions.

        The parts suitable for delexicalization are located from the data field dialogue_acts and its fields span_end and span_start,
        which are replaced with the corresponding slot names from act_slot_name enclosed in brackets,
        e.g., [name] or [pricerange].

        A data field dialogue_act has the following structure:
            "dialog_act": {
            (... other fields)
                "span_info": {
                    "act_slot_name": ["area"],
                    "act_slot_value": ["east"],
                    "act_type": ["Attraction-Inform"],
                    "span_end": [39],
                    "span_start": [35]
                }
            }
        """
        slot_info = dialogue_act.get("span_info", {})

        starts = slot_info.get('span_start', [])
        ends = slot_info.get('span_end', [])
        names = slot_info.get('act_slot_name', [])

        # unmodified
        if not starts or not ends or not names:
            return utterance

        # Sort spans by start index to ensure proper replacement order
        spans = sorted(zip(starts, ends, names), key=lambda x: x[0])

        # Build the delexicalized utterance
        delex = ""
        last_end = 0

        try:
            for start, end, name in spans:
                delex += utterance[last_end:start] + f"[{name}]"
                last_end = end  # only from last index
            
            # last remaining text
            delex += utterance[last_end:]

            return delex
        except:
            # unchanged
            return utterance

    def parse_dialogue_into_examples(self, dialogue, context_len=None):
        """
        Parses a dialogue into a list of examples. Each example includes the preceding context of max. length context_len
        and the corresponding response (both the original response and a delexicalized version, where slot values are
        replaced by placeholders with slot names).

        Each is a dictionary of the folowing structure:
        {
            # for HW2:
            'context': list[str],  # list of context_len utterances preceeding the current utterance
            'utterance': str,  # the string with the current response
            'delex_utterance': str,  # the string with the current response which is delexicalized, i.e. slot values are
                                     # replaced by corresponding slot names in the text.
        }
        """
        examples = []

        # user and system utterances with annotations for dialogue acts and user belief states
        turns = dialogue['turns']
        # list of all preceding utterances
        context = []

        # only system responses
        for i in range(0,len(turns["turn_id"])):
            if (turns["speaker"][i] == 1):
                utterance = turns["utterance"][i]
                delex_utterance = self.delexicalize(turns["utterance"][i], turns["dialogue_acts"][i])
                    
                # build preceding context as we go
                # each time we add the previous utterance
                if (i >0):
                    context.append(turns["utterance"][i - 1])

                # context_len utterances preceding the current utterance
                context_i = context[:] if context_len is None else context[max(0, len(context) - context_len):]

                # dictionary entry
                examples.append({
                    'context': context_i,
                    'utterance': utterance,
                    'delex_utterance': delex_utterance,
                })

        return examples
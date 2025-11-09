import json
import os
import random
import torch
import numpy as np
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from diallama.mw_loader import Dataset
from diallama.generate import GenerationWrapper


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

HF_TOKEN = ""

def process_data(split, domain, context_len, output_fname):

      dataset = Dataset(split, domain=domain, context_len=context_len)
      data_loader = DataLoader(dataset,
                              batch_size=5,
                              shuffle=True,
                              collate_fn=lambda x: x)  # collate_fn is a no-op

      # (HW2): Load the model and tokenizer
      from transformers import AutoModelForCausalLM, AutoTokenizer

      tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=HF_TOKEN)
      model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=HF_TOKEN)

      # Initialize the generator with an empty system and response prompt
      generator = GenerationWrapper(model, tokenizer, system_prompt='', response_prompt='')

      # We don't need to care about prompts yet
      generator = GenerationWrapper(model, tokenizer, system_prompt='', response_prompt='')

      with open(os.path.join(os.path.dirname(__file__), output_fname), "w+") as f:
            for idx, batch in enumerate(data_loader, ):
                  print(f"Batch {idx} normal:", file=f)
                  print(json.dumps(batch, indent=2), file=f)
                  collate_batch = generator.collate_fn(batch)
                  print(f"Batch {idx} collate:", file=f)
                  print(collate_batch, file=f)
                  print(f"Number of tokens in batch {idx}:",
                        [sum([len(utt.split()) for utt in x['context']]) + len(x['utterance'].split()) for x in batch],
                        file=f)
                  print(f"Number of response tokens in batch {idx}:",
                        [len(x['utterance'].split()) for x in batch],
                        file=f)
                  print(f"Number of response tokens in batch {idx} after collate:",
                        [len(resp) for resp in collate_batch["response"]],
                        file=f)
                  break

            dataset = Dataset("validation", domain=domain, context_len=0)
            print(f"Valid num samples for {domain}", len(dataset), file=f)
            dataset = Dataset("test", domain=domain, context_len=0)
            print(f"Test num samples for {domain}", len(dataset), file=f)


if __name__ == "__main__":

      ap = argparse.ArgumentParser()
      ap.add_argument("--split", type=str, default="test", help="Which data split to use")
      ap.add_argument("--domain", type=str, default="hotel", help="Which domain to filter")
      ap.add_argument("--context_len", type=int, default=3, help="Maximum number of dialogue turns to include in the context")
      ap.add_argument("--output", type=str, default="output.txt", help="Output file")
      args = ap.parse_args()
      process_data(args.split, args.domain, args.context_len, args.output)

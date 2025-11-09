
import random
import torch
import numpy as np
from logzero import logger

from diallama.generate import GenerationWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

# get default settings for generation (you may adjust these if needed)
from hw3.test import INPUTS, GENERATION_CONFIGS
from hw4.train import MODEL_PATH, SYSTEM_PROMPT, RESPONSE_PROMPT

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

if __name__ == "__main__":

    # Load the pretrained model and tokenizer
    logger.info('Loading model and tokenizer...')
    HF_TOKEN = ""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, token=HF_TOKEN)

    print("Model: ", model)
    print("Tokenizer: ", tokenizer)

    print("*" * 80)
    print(f"System prompt: {SYSTEM_PROMPT}")
    print(f"Response prompt: {RESPONSE_PROMPT}")

    generator = GenerationWrapper(model, tokenizer, system_prompt=SYSTEM_PROMPT, response_prompt=RESPONSE_PROMPT)

    print(f"Generation config: {str(GENERATION_CONFIGS[0])}")
    print("*" * 80)

    # Test all prompts & settings on the set of inputs above
    for input_text in INPUTS:
        print(f"Input: {input_text}")
        response = generator.generate_response([input_text], GENERATION_CONFIGS[0])
        print(f"Response: {response}")
        print()

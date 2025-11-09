#google colab exec
import sys
#sys.path.append('/content/rodrigad')
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random

import torch
import numpy as np
from logzero import logger

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from transformers import get_scheduler
from diallama.mw_loader import Dataset
from diallama.trainer import Trainer
from diallama.generate import GenerationWrapper

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

CTX_LEN = 5
BS = 1  # batch size, increase if your GPU RAM allows
EPOCHS = 1

# store trained model here
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model'))

SYSTEM_PROMPT = ("You are a helpful agent specialized in assisting users with hotel reservations, travel planning, trips... " + 
                "Your responses must be relevant, accurate and helpful to satisfy the user's request. ")
RESPONSE_PROMPT = "(NICE RESPONSE) Be sensible and address user's concerns as if you were a lovely old travel agent, giving insightful tips for accommodation in the area."

if __name__ == "__main__":

    logger.info('Loading model and tokenizer...')

    # HW4: Load model and tokenizer
    HF_TOKEN = "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"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=HF_TOKEN)

    # load the data
    logger.info('Loading data...')
    train_dataset, valid_dataset = Dataset('train', context_len=CTX_LEN), Dataset('validation', context_len=CTX_LEN)

    logger.info('Initializing formatting & batching')
    genwrapper = GenerationWrapper(model, tokenizer, system_prompt=SYSTEM_PROMPT, response_prompt=RESPONSE_PROMPT)

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, collate_fn=genwrapper.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BS, shuffle=False, collate_fn=genwrapper.collate_fn)

    logger.info('Initializing PEFT model, optimizer & scheduler...')
    # (HW4): initialize the LoRA settings for the PEFT model variant
    # You can use https://kickitlikeshika.github.io/2024/07/24/how-to-fine-tune-llama-3-models-with-LoRA.html as a reference
    # for the LoRA config (see point 6 in the blog post). Feel free to use the default setting from there.

    # In general:
    # - we want the TaskType to be CAUSAL_LM, since that's what we're doing (=next word generation)
    # - high rank means wider matrixes, i.e. more parameters (more power), but more memory/less speed
    # - high alpha means more weight to the LoRA parameters, but you don't want to overpower the original model
    # - you want some low dropout
    # - LoRA should ideally be applied to the linear projections in the model (target_modules)
    # default config, from original LORA paper: using lora, we should target the linear layers only
    lora_config = LoraConfig(
        r=32,  # rank for matrix decomposition
        lora_alpha=16,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj"
        ],
        lora_dropout=0.05,
        bias='none',
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # (HW4): initialize the optimizer and scheduler. You can go with any reasonable defaults, e.g. AdamW and a linear, cosine
    # or even constant scheduler. See https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
    optimizer = AdamW(model.parameters(), 
                      lr=1e-5, # lowered from 4 to 5 
                      weight_decay=1e-2)

    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_scheduler("linear", 
                              optimizer=optimizer, 
                              num_training_steps=num_training_steps,
                              num_warmup_steps=num_warmup_steps)

    # this will create our trainer object using all info we gathered so far & start training
    trainer = Trainer(
        model,
        train_loader,
        valid_loader,
        EPOCHS,
        optimizer,
        scheduler
    )
    logger.info('Starting training...')
    trainer.train()

    logger.info(f'Merging and saving model to {MODEL_PATH}')

    # Note: We're merging LoRA weights into the trained model here -- this'll make the resulting model faster
    # See https://huggingface.co/docs/peft/main/en/developer_guides/lora#merge-lora-weights-into-the-base-model for details.
    model.merge_and_unload()

    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

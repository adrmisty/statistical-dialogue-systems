from typing import Tuple, Optional

import tqdm
from transformers import PreTrainedModel
from torch.optim import Optimizer, lr_scheduler
from logzero import logger
import torch

from .mw_loader import Dataset


class Trainer:

    def __init__(self,
                 model: PreTrainedModel,
                 train_data_loader: Dataset,
                 valid_data_loader: Dataset,
                 epochs: int,
                 optimizer: Optimizer,
                 scheduler: lr_scheduler._LRScheduler):
        self.model = model
        self.device = model.device
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        assert epochs > 0
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self):
        """
        Train the model for self.epochs epochs using the training data loader, the optimizer
        and scheduler given in the constructor.
        """
        logger.info('Starting training...')
        for epoch in range(self.epochs):
            logger.info(f'====== Epoch {epoch}/{self.epochs} Training ======')
            self.model.train()  # make sure we're in training mode (eval turns it off after each epoch)
            for step, batch in enumerate(tqdm.tqdm(self.train_data_loader)):
                """
                The batch should include the following (see generate.py):
                (also include utterances, delex utterances...)
                {
                    "concatenated": Tensor[bs, maxlen],
                    "attention_mask": Tensor[bs, maxlen],
                    "response_mask": Tensor[bs, maxlen],
                }
                """
                # (HW4): mplement model forward and backward steps and all the necessary related logic
                # (zero_grad, optimizer step, etc.)
                # Feed the model with the concatenated prompt + response and the attention mask,
                # with target labels being the same as response tokens (they'll be shifted automatically).
                # Using the response mask, only compute the loss on the response tokens (not the prompt tokens),
                # i.e. don't learn to generate the prompt -- set the target to -100 for the prompt tokens
                # (see ignore_index under https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

                input_ids = batch["concatenated"]
                at_masks = batch["attention_mask"]
                re_masks = batch["response_mask"]

                if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
                    print("> NaN or Inf values found in input_ids!")


                # only train the model to generate the response, not the context, 
                # by setting the model's target labels properly. 
                # Make use of the response_mask to produce the correct labels input.
                if input_ids.shape != re_masks.shape:
                    re_masks = re_masks[:, :input_ids.shape[1]]

                labels = input_ids.clone()
                labels[~re_masks.bool()] = -100  # to be ignored

                # feed the whole concatenated tensors into the model as input_ids, including the context.
                # don't forget to use the attention_mask, so you avoid performing attention over padding
                output = self.model(input_ids, attention_mask=at_masks, labels=labels)
                loss = output.loss

                if torch.isnan(loss).any():
                    print(f"> NaN loss detected at step {step}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if step % 100 == 0:
                    logger.info(f'Step {step}: loss = {loss.item()}')

            logger.info(f'======= Epoch {epoch}/{self.epochs} Validation ===========')
            self.eval()

    def eval(self, data_loader: Optional[Dataset] = None) -> Tuple[float, float]:
        """
        Compute perplexity and token accuracy on the validation set (by default) or any set given in the parameter.
        :param data_loader: DataLoader, data loader to use for evaluation (default is self.valid_data_loader)
        :return: Tuple[float, float], perplexity and token accuracy
        """
        # set evaluation mode (for dropout, batchnorm, etc.)
        self.model.eval()

        # go with the validation set by default
        if data_loader is None:
            data_loader = self.valid_data_loader

        # (HW4): mplement evaluation step + Token accuracy & perplexity computation
        # 1) Generate outputs for all batches (just 1 forward pass, i.e. next-token prediction,
        #        not full autoregressive generation)
        # 2) Compare the argmax of output tokens to the target response tokens
        #        (i.e., to input tokens shifted by 1) to compute token accuracy (remember to ignore padding!).
        # 3) Compute perplexity (note it's quite easy since we're using cross entropy loss)
        
        total_loss = 0 # accumulated loss
        correct_tokens = 0
        total_tokens = 0

        with torch.no_grad():  # no gradient computation
            for batch in tqdm.tqdm(data_loader, desc="Evaluating"):
                # Move data to the device
                input_ids = batch["concatenated"]
                attention_mask = batch["attention_mask"]
                response_mask = batch["response_mask"]

                labels = input_ids.clone()
                labels[~response_mask.bool()] = -100  # Ignore prompt tokens by setting them to -100

                # pass inputs to the model
                output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = torch.clamp(output.logics, min=1e-8)
                loss = output.loss

                # token accurazy
                predictions = logits.argmax(dim=-1) 
                correct_tokens += ((predictions == labels) & response_mask.bool()).sum().item()
                total_tokens += response_mask.sum().item()

                total_loss += loss.item() * input_ids.size(0)

        avg_loss = total_loss / len(data_loader.dataset)
        perplexity = torch.exp(torch.tensor(avg_loss))  # perplexity = exp(loss)
        valid_perplexity = perplexity.item()
        valid_token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0

        # log and return results
        logger.info(f'perplexity: {valid_perplexity}')
        logger.info(f'token acc: {valid_token_acc}')
        return valid_perplexity, valid_token_acc

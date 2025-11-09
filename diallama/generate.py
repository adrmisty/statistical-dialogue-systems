from typing import Text, List
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig
from diallama.database import MultiWOZDatabase
import json, re
import torch
import random

class GenerationWrapper:

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 system_prompt: Text,
                 response_prompt: Text,
                 dst_prompt: Text):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
        self.system_prompt = system_prompt
        self.response_prompt = response_prompt

        # HW5
        self.database = MultiWOZDatabase()
        self.dst_prompt = dst_prompt

    def collate_fn(self, batch, stage):
        """
        Tokenize and format the examples in the batch for use with the model, including all prompts.
        The input batch is stuff returned by the Dataset class (list of examples, with text fields
        of the contexts and the responses).

        For HW2, the output should include the following:
        - 'response': list[list[int]] - tokenized corresponding responses
        - 'delex_response': list[list[int]] - tokenized corresponding delexicalized responses

        For HW3, include this in addition to the above:
        - 'prompt': list[list[int]] - tokenized and correctly formatted prompts, including the
                    system message, instructions and the context

        For HW4, include this:
        - 'concatenated': 2D-Tensor[int] - tokenized and correctly formatted concatenated
            prompt + response (delexicalized by default), dimensions: batch size x total number of tokens
            Use max. number of tokens in the batch and pad shorter sequences with the padding token (=EOS).
        - 'attention_mask': 2D-Tensor[int] - mask for the concatenated prompt + response -- same dimensions as
                    "concatenated", with all 1's for valid tokens and 0's for padding.
        - 'response_mask': 2D-Tensor[int] - mask for the prompt tokens, same dimensions as "concatenated",
                    set to 1 for response tokens, 0 for prompt tokens or padding.

        For HW5, we add a new parameter 'stage' which determines whether the function prepares inputs for dialogue state tracking, 
        response generation, or selects randomly for each batch example.
        - stage='dst': 
            1. use self.dst_prompt with the context to create the prompt. 
            2. assume batch contains a dialogue_state (a slot-value dict, e.g., {'area': 'north', 'stars': '3'}), 
            used as both response and delex_response.
            3. adjust concatenated, attention_mask, and response_mask accordingly.
            4. Ensure functionality when dialogue_state is empty or missing.
        - stage='response':
            1. use self.response_prompt, incorporating context, dialogue_state, and db_results (e.g., a count or hotel entries).
            2. Handle dialogue_state as above, even if initially empty.
            3. Otherwise, process similarly to dst.
        - stage='mixed':
            1. randomly select between 'dst' and 'response' for each batch example.
        """

        # batch: list of dicts with keys 'context', 'utterance' and 'delex_utterance'.
        # context: a list of strings
        # utterance and delex: strings without and with replaced slots respectively.

        # (HW2): Tokenize both versions of the utterances and return as 'response' and 'delex_response'.
        tokenized_responses = []
        tokenized_delex_responses = []

        # (HW3): Prompting
        tokenized_prompts = []
        attention_masks = []

        # (HW4): Concatenations
        concat = []

        for input_convo in batch:

            if isinstance(input_convo, dict):
                input_convo = [input_convo]

            # 'context' (preceding dialogue turns)
            # to be used in case there is no (delex) utterance

            if 'context' in input_convo[0] and input_convo[0]['context'] is not None:
                turns = input_convo[0]['context']
                roles = ["user", "assistant"]  # Alternating roles

                context = []
                context_text = ""
                for i, turn in enumerate(turns):
                    role = roles[i % 2]  # alternate
                    content = turn.strip()
                    context.append({"role": role, "content": content})
                    context_text += f"{role}: {content} "

                tokenized_context = self.tokenizer.encode(
                    context_text,
                    add_special_tokens=True
                )
            else:
                tokenized_context = []  # Empty if no context is present

            if stage == 'dst':
                dialogue_state = input_convo[0].get('dialogue_state', {})

                response_text = str(dialogue_state) if dialogue_state else "{}"  # str representation of the dict
                tokenized_response = self.tokenizer.encode(response_text, add_special_tokens=True)
                tokenized_delex_response = tokenized_response  # identical to response

                prompt_text = f"{self.dst_prompt}\n\n{context_text}"

            elif stage == 'response':
                dialogue_state = input_convo[0].get('dialogue_state', {})
                db_results = input_convo[0].get('db_results', "")
                response_text = input_convo[0].get('utterance', "")
                delex_response_text = input_convo[0].get('delex_utterance', response_text)

                tokenized_response = self.tokenizer.encode(response_text, add_special_tokens=True)
                tokenized_delex_response = self.tokenizer.encode(delex_response_text, add_special_tokens=True)

                prompt_text = (
                    f"{self.system_prompt}\n{self.response_prompt}\n"
                    f"{context_text}\nState: {dialogue_state}\nDB: {db_results}\n"
                )

            elif stage == 'mixed':
                whatever = random.choice(['dst', 'response']) # randomly choose between the 2
                return self.collate_fn(input_convo, whatever) #recursive

            else:
                raise ValueError("> Non-valid state, must be : 'dst', 'response', 'mixed'!!.")

            # HW4: special tokens
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>") # forgot it is eot_id instead of eot hehe
            end_of_text_id = self.tokenizer.eos_token_id
            
            # prompt text is correct (contains prompt and context, exchange of user/system turns, and DST prompting)
            tokenized_prompt = self.tokenizer.encode(
                prompt_text,
                add_special_tokens=True
            )  # should include attention mask

            tokenized_responses.append(tokenized_response)
            tokenized_delex_responses.append(tokenized_delex_response)
            tokenized_prompts.append(tokenized_prompt)

            # HW4: all in one list, HW5: concatenated
            c = tokenized_prompt + [eot_id] + tokenized_response + [end_of_text_id]
            # found some None in the sequence
            # replace with empty sequence
            if None in c:
                c = [self.tokenizer.pad_token_id]

            concat.append(c)


        # (HW4): attention masks and padding
        max_padding = max(len(seq) for seq in concat)
        padded_concat = []
        response_masks = []

        for c in concat:
            # pad sequences up to max length
            to_pad = max_padding - len(c)
            padded = c + [self.tokenizer.pad_token_id] * to_pad
            padded_concat.append(padded)

            # attention mask: 1 for valid tokens, 0 for padding
            attention_mask = [1] * len(c) + [0] * to_pad
            attention_masks.append(attention_mask)

            # response mask: 1 for response tokens only, 0 otherwise
            response_mask = (
                [0] * (len(tokenized_context) + 1)  # content and eot_id
                + [1] * len(tokenized_response)     # response
                + [0]                               # end_of_text
                + [0] * to_pad                      # padding
            )
            response_masks.append(response_mask)

        # (HW4): Required info into tensors
        concat_tensor = torch.tensor(padded_concat, dtype=torch.long)
        at_masks_tensor = torch.tensor(attention_masks, dtype=torch.bool)
        re_masks_tensor = torch.tensor(response_masks, dtype=torch.bool)

        output = {
            "prompt": tokenized_prompts,  # hw3
            'response': tokenized_responses,
            'delex_response': tokenized_delex_responses,
            "attention_mask": at_masks_tensor,  # Tensor version
            "concatenated": concat_tensor,  # hw4
            "response_mask": re_masks_tensor,
            "dialogue_state": dialogue_state # hw5
        }

        return output

    def generate_response(self, context: List[Text], generation_config: GenerationConfig):
        """Generate a response to the given context.
            
            Note: this context is made up of single-turn dialogue inputs.
            
            For HW5, we implement two-stage generation. It shall return a tuple: "", {}, [] 
            - The LLM's final response (as text)
            - The parsed dialogue state (as a dict)
            - The database results (raw output of self.database.query, i.e. the matching database entries)
        """
        self.model.eval()
        # padding by end-of-string token (not important without batches, but will suppress HuggingFace warnings)
        generation_config.pad_token_id = self.tokenizer.eos_token_id

        # (HW5): two-stage generation
        try:
            # Stage 1: Dialogue State Tracking (DST)
            # use collate_fn in the dst mode
            batch=[{'context': context,
                'utterance': '',
                'delex_utterance': '',
                'dialogue_state': {'area':"", 'price': 0, 'features':[]},
                'db_results': 0,
                }],

            dst_output = self.collate_fn(batch, stage='dst')
            dst_input_ids = dst_output['concatenated']
            dst_attention_mask = dst_output['attention_mask']

            # call LLM with the prepared input to get the “response”, i.e., the current dialogue state
            dst_model_output = self.model.generate(
                input_ids=dst_input_ids,
                attention_mask=dst_attention_mask,
                **generation_config.to_dict()
            )

            # extract dialogue state
            # DST is structured as a JSON so we try to parse it as a JSON object
            # otherwise just give an empty JSON state
            dst_decoded = self.tokenizer.decode(dst_model_output[0], skip_special_tokens=True)
            
            # all valid matches
            json_matches = re.findall(r"\{.*?\}", dst_decoded, flags=re.DOTALL)
            # find the one that is formed as a valid json object
            for match in json_matches:

                try:
                    json_str = match.group()
                    json_str = json_str.replace("'", '"')
                    dialogue_state = json.loads(json_str)
                    break

                except (SyntaxError, NameError):
                    continue

            if not dialogue_state:
                print("> Warning: malformed dialogue state!")
                return "", {}, []  # error
            print(dialogue_state)

            # query DB for extracted state
            # call self.database.query() with the domain set to hotel and 
            # the constraints using a slot-value dict as shown above (JSON DST format)
            db_results = self.database.query(domain="hotel", constraints=dialogue_state)

            # Stage 2: 
            # if everything goes well, u can use that state now for collat
            # include dialogue_state, db-results and pass into batch for response 
            batch = [{'context': context, 'utterance': '', 'delex_utterance': '', 'dialogue_state': dialogue_state, 'db_results': db_results}]
            output = self.collate_fn(batch, 'response')
            # 2. both input_ids (tokenized prompt) and attention mask should be tensors (w shape)
            input_ids = output['concatenated']
            attention_mask = output["attention_mask"]

            model_output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config.to_dict()
            )

            # empty response (not state or db)
            if dst_model_output is None or dst_model_output.size(0) == 0:
                return "", dialogue_state, db_results 

            # 3. response (which includes formatted input in the model's output)
            decoded = self.tokenizer.decode(model_output[0], skip_special_tokens=True) # skipspecialtokens=true
            decoded = decoded[len(decoded):]  # only return the model output, don't repeat the prompt
            prompt_decoded = self.tokenizer.decode(decoded[0], skip_special_tokens=True)

            # return only the model's response, not the whole thing
            generated_response = decoded.replace(prompt_decoded, "")
            generated_response = generated_response.replace("assistant", "").strip()
        
            return generated_response.strip(), dialogue_state, db_results
    
        # "gracefully" handle exceptions
        except Exception as e:
            print(f"> (!) Error during response generation: {e}")
            return "", {}, [] # empty response
            
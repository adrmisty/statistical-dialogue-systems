
from diallama.generate import GenerationWrapper
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

# get all our settings from hw5
from hw5.test import SYSTEM_PROMPT, DST_PROMPT, RESPONSE_PROMPT, GENERATION_CONFIG


if __name__ == "__main__":

    # TODO Load the model and tokenizer
    model = AutoModelForCausalLM(...)
    tokenizer = AutoTokenizer(...)

    generator = GenerationWrapper(model, tokenizer, system_prompt=SYSTEM_PROMPT, dst_prompt=DST_PROMPT, response_prompt=RESPONSE_PROMPT)

    test_set = Dataset('test', domain='hotel', context_len=3)
    data_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=generator.collate_fn)

    # TODO store outputs in a structured form
    # TODO run evaluation on them
    for batch in data_loader:
        response, dst, db_results = generator.generate_response(batch, GENERATION_CONFIG)
        ...

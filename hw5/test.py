#google colab exec
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diallama.generate import GenerationWrapper
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer


# These are the input data you should use for testing (now they're multi-turn)
output_fname = "./hw5/output.txt"
INPUTS = [
["I want a cheap place to stay in the north", "Do you require any features?", "Yes, free parking and internet."],
    ["I'm looking for a hotel in the centre.", "Do you have a preferred star range.", "I want a 3-star hotel."],
    ["I need a bed and breakfast that has parking.", "Any preference on the area?", "I want something in the east."],
    ["Give me a five-star hotel.", "Sure, do you have a preference on the price range?", "No, I don't care."],
    ["I want some cheap hotel to stay for four nights."],
    ["Hi! I'm looking for a hotel, can you help me?"],
    ["I want a hotel in the city center.", "Any price preference?", "No, show me what's available", "OK, what date would that be?", "Three nights from Friday."],
]

# Write the system prompt ("You are a hotel assistant..."), the response prompt (i.e., instruction to respond in a dialogue)
#      where you now include the dialogue state and database results, as well as as the newly created dialogue state tracking (DST)
#      prompt, where the system is asked to infer the dialogue state (slot-wise use preferences) from the context
SYSTEM_PROMPT = ("\nYou are a helpful agent specialized in assisting users with hotel reservations, travel planning, trips... " + 
                "Your responses must be relevant, accurate and helpful to satisfy the user's request. ")

RESPONSE_PROMPT = (
        "\nBe sensible and address user's concerns as if you were a lovely old travel agent, giving insightful tips for accommodation in the area. Be as concise as possible"
    )

DST_PROMPT = (
    "Extract the user's preferences from the given conversation context. "
    "Represent the output strictly as a JSON object with these keys: area, price and features. "
    "Do not include any additional text in the output."
)

# Prepare the generation config, feel free to use a relatively basic one, make sure you pass in the DST prompt as well as response prompt
#      Keep the max_new_tokens limited to a reasonable number (<100), so the generation doesn't run too long
GENERATION_CONFIG = GenerationConfig(
        max_new_tokens=90,  # Greedy search, simplest decoding alg
        temperature=.7,
        num_beams=1,        # Single beam (greedy)
    )


if __name__ == "__main__":

    # Load the model and tokenizer
    HF_TOKEN = ""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=HF_TOKEN)
    
    # ////// EMPTY STATE start

    generator = GenerationWrapper(model, tokenizer, system_prompt=SYSTEM_PROMPT, dst_prompt=DST_PROMPT, response_prompt=RESPONSE_PROMPT)
    """
    tokenizer.decode(generator.collate_fn(
        batch=[{'context': INPUTS[0],
                'utterance': '',
                'delex_utterance': '',
                'dialogue_state': {'area':"", 'price': None, 'features':[]},
                'db_results': 0}],
        stage='dst')["prompt"][0])
    
    tokenizer.decode(generator.collate_fn(
        batch=[{'context': INPUTS[0],
                'utterance': '',
                'delex_utterance': '',
                'dialogue_state': {'area':"", 'price': None, 'features':[]},
                'db_results': 0,
                }],
        stage='response')["prompt"][0])
    """
    # ////// EMPTY STATE end


    # Test all prompts & settings on the set of inputs above
    # Store & analyze the result

    with open(output_fname, 'a') as of:

        asterisks = "*" * 80
        separator = "_" * 40

        to_output = f"\n{asterisks}\nSystem prompt: {SYSTEM_PROMPT}\n{asterisks}"
        to_output += f"\n\n\n{asterisks}\nResponse prompt: {RESPONSE_PROMPT}"
        to_output += f"\nGeneration config: {str(GENERATION_CONFIG)}\n"

        to_output += f"\n{separator}\n"

        for input_context in INPUTS:
            response, dialogue_state, database_results = generator.generate_response(input_context, GENERATION_CONFIG)

            # save everything to file
            to_output += f"Input: " + "\n".join(input_context)
            to_output += f"\nDialogue state: {dialogue_state}"
            to_output += f"\n# DB results: {len(database_results)}"
            to_output += f"\nResponse: {response}"
            to_output += f"\n{asterisks}"
                  
            # save output & print it
            of.write(to_output)
            print(to_output) # comment when i don't want to debug anymore
            to_output = ""          

        of.close()


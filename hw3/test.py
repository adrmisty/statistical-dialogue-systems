#google colab exec
import sys
#sys.path.append('/content/rodrigad')
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diallama.generate import GenerationWrapper
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer


# These are the input data you should use for testing (as single-turn dialogues only, we'll get to multi-turn later)
INPUTS = [
        ["Hello, I would like to find a hotel in Madrid.",
        "Sure! What dates will you be staying, and do you have any preferences for location?",
        "I’ll be there from December 30th to January 1st to spend New Year's. I’d like a place near the Plaza del Sol.",
        "Noted! Do you have any preference for budget or any preference to take into account?\n",
        "My budget is 100€ a night tops, and I definitely do not want to share a room in a hostel. Would it also be possible to have sauna?\n\n"],
        [
        "I’m planning a trip to Lisbon. Can you help me find an accessible hotel?",
        "Of course! Are there specific features you’re looking for, such as an elevator, step-free access, or acceptance of guide dogs?"
        "Yes, I’d like a hotel without steps and/or stairs, an elevator and a central location near major attractions."
        ],
        [
        "Hello, I am an Orthodox Jew planning a trip to New York. I would like to stay in an inclusive hotel that allows me to celebrate Shabbat."
        "Sure! Would you like for the hotel to provide kosher dining?"
        "Yes, I am looking for kosher options and in vicinity of synagogues. My budget is 200$ a night."
        "Noted! Would you like full dining, or just breakfast?"
        "I would be fine with just dinner, but it should have good kosher food places for breakfast and lunch nearby as well."
        ]
]

# (HW3): Write the system prompt ("You are a hotel assistant...") and the response prompt (i.e., instruction to respond in a dialogue)
#      Try out at least 3 different response prompts.
SYSTEM_PROMPT = ("You are a helpful agent specialized in assisting users with hotel reservations, travel planning, trips... " + 
                "Your responses must be relevant, accurate and helpful to satisfy the user's request. ")

RESPONSE_PROMPTS = [
    # Prompt 1: General for training
    (
        "(GENERAL RESPONSE)\nResponse must be done in a generalizable manner suitable for training a dialogue system, \nso you must avoid specific place names " +
        "but provide information structured \naccording to type of accommodation, typical features,\n price ranges... in different subsections."

    ),
    # Prompt 2: Specific with format
    (
        "(FORMATTED RESPONSE)\nResponse with specific recommendations following this format:\n" +
            "1. Provide two or three recommendations with brief descriptions, focusing on required facilities.\n"
            "2. Mention pricing and availability information (if applicable).\n"
            "3. Include travel tips that might be helpful for the user specifically.\n\n"

    ),
    # Prompt 3: Friendly travel agent :D
    (
        "(NICE RESPONSE <3)\nBe sensible and address user's concerns as if you were a lovely old travel agent,\ngiving insightful tips for accommodation in the area."
    )
]

# Prepare generation config
#      Try three different generation strategies (e.g. greedy/beam search/top-k sampling/nucleus sampling) for one of your prompts
#      Make sure you use max_new_tokens limited to a reasonable number (<100), so the generation doesn't run too long
#       https://huggingface.co/blog/how-to-generate
GENERATION_CONFIGS = [
    # Greedy search
    # - simplest decoding method
    # - highest probability: argmaxP(w|w1:t-1)
    GenerationConfig(
        max_new_tokens=90,
        do_sample=False,    # Disable sampling
        num_beams=1,        # Single beam (greedy)
    ),
]

"""
    # Beam search
    # - avoids missing hidden high probability sequences
    # - keeping the most likely num_beams of hypotheses at each time step
    # - eventually chooses the highest probability one
    GenerationConfig(
        max_new_tokens=90,
        do_sample=False,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    ),

    # Top-k sampling
    # - eliminate weirdest candidates and keep K most likely next words with their probability mass redistributed
    # - make probability distribution sharper (increase prob. of most likely, decrease of least likely) 
    #   by adjusting temperature of the softmax function
    GenerationConfig(
        max_new_tokens=90,
        do_sample=True,     # Enable sampling
        top_k=50,           # Sample from top 50 tokens
        temperature=0.6,    # Adjust randomness
    ),

    # Top-p sampling
    # - chooses from the smallest possible set of words whose cumulative probability exceeds the probability p
    #   basically choose the minimum word set whose probability exceed 90%
    GenerationConfig(
        max_new_tokens=90,
        do_sample=True,
        top_k=50,
        top_p=0.9, # cumulative probability
        num_return_sequences=3,
    ),
"""



if __name__ == "__main__":

    HF_TOKEN = ""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=HF_TOKEN)
    output_fname = "./hw3/output-good-final.txt"


    # checking that prompt is properly formatted
    # generator = GenerationWrapper(model, tokenizer, system_prompt=SYSTEM_PROMPT, response_prompt=RESPONSE_PROMPTS[0])
    #output = tokenizer.decode(generator.collate_fn([{'context': [INPUTS[0]], 'utterance': '', 'delex_utterance': ''}]))
    #input_prompt = f"{asterisks}\nInput prompt:\n{output['prompt'][0]}"
    #to_output = input_prompt
    #print(f"{asterisks}\nPrompt formatting check\n{asterisks}")
    #print(input_prompt)


    with open(output_fname, 'a') as of:

      asterisks = "*" * 80
      separator = "_" * 40
      syst_prompt= f"\n{asterisks}\nSystem prompt: {SYSTEM_PROMPT}\n{asterisks}"
      to_output = syst_prompt

      for response_prompt in RESPONSE_PROMPTS:

          resp_prompt = f"\n\n\n{asterisks}\nResponse prompt: {response_prompt}"
          to_output += resp_prompt
          #print(resp_prompt)

          generator = GenerationWrapper(model, tokenizer, system_prompt=SYSTEM_PROMPT, response_prompt=response_prompt)

          for generation_config in GENERATION_CONFIGS: # only test 1
              gen_conf = f"\nGeneration config: {str(generation_config)}\n"
              to_output += gen_conf
              #print(gen_conf)

              # Test all prompts & settings on the set of inputs above
              #print(f"\n{separator}")
              to_output += f"\n{separator}"

              for input_text in INPUTS:
                  
                  output = generator.generate_response([input_text], generation_config)
                  
                  resp_test = f"{output}\n{separator}"
                  to_output += resp_test
                  #print(resp_test)
                  
                  # update file
                  of.write(to_output)
                  to_output = ""          
    
      of.close()
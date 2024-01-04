import os
from gpt4all import GPT4All
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file
WOLFRAM_APP_ID = os.getenv("WOLFRAMPLHA_APP_ID")
MODELS_PATH = os.getenv("MODELS_PATH")

class LLM:
    def __init__(self, name, prompt_template):
        self.name = name
        self.prompt_template = prompt_template
        
    def create_full_prompt(self, actual_prompt):
        parts = self.prompt_template.split('\n')
        parts[2] = actual_prompt
        full_prompt = '\n'.join(parts)
        return full_prompt
           
        
model1 = LLM("mistral-7b-openorca.Q4_0.gguf","### Human:\n\n### Assistant:\n")
model2 = LLM("orca-2-7b.Q4_0.gguf","### Human:\n\n### Assistant:\n")
model3 = LLM("gpt4all-falcon-q4_0.gguf","### Instruction:\n\n### Response:\n")

llms = [model1, model2, model3]
extra_instructions = "Your answer must consist a few words maximum."
for llm in llms:
    model = GPT4All(llm.name, MODELS_PATH)
    full_prompt = llm.create_full_prompt("What's the highest mountain in the world? " + extra_instructions)
    output = model.generate(full_prompt, max_tokens=5)
    print(f"{llm.name} answered:\n{output}")
    full_prompt = "Who was the USA president in 1967?"
    output = model.generate(full_prompt, max_tokens=5)
    print(f"{llm.name} answered:\n{output}")
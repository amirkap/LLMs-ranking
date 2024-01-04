import os
import requests
import csv 
from llm import LLM
from gpt4all import GPT4All
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file
WOLFRAM_APP_ID = os.getenv("WOLFRAMPLHA_APP_ID")
MODELS_PATH = os.getenv("MODELS_PATH")

           
def wolfram_alpha_short_answer_query(query, app_id):
    base_url = "https://api.wolframalpha.com/v1/result"
    encoded_query = "+".join(query.split())  # URL encode the query
    url = f"{base_url}?i={encoded_query}&appid={app_id}"

    response = requests.get(url)

    if response.status_code == 200:
        return response.text.strip()
    else:
        return f"Error {response.status_code}: {response.text}"

query = "Who directed the movie 'Jaws'?"
result = wolfram_alpha_short_answer_query(query, WOLFRAM_APP_ID)
print(result)
        
model1 = LLM("mistral-7b-openorca.Q4_0.gguf","### Human:\n\n### Assistant:\n")
model2 = LLM("orca-2-7b.Q4_0.gguf","### Human:\n\n### Assistant:\n")
model3 = LLM("gpt4all-falcon-q4_0.gguf","### Instruction:\n\n### Response:\n")

llms = [model1, model2, model3]
extra_instructions = ""
for llm in llms:
    model = GPT4All(llm.name, MODELS_PATH)
    full_prompt = llm.create_full_prompt(query + extra_instructions)
    print(full_prompt)
    output = model.generate(full_prompt)
    print(f"{llm.name} answered:\n{output}")
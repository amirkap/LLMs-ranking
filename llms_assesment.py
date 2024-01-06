import os
import requests
import csv 
import redis
import time
import pandas as pd
from llm import LLM
from gpt4all import GPT4All
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file
WOLFRAM_APP_ID = os.getenv("WOLFRAMPLHA_APP_ID")
MODELS_PATH = os.getenv("MODELS_PATH")

KEY_TTL = 14400 # Redis key time-to-live is 4 hours
redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)
# redis_client.setex('ooo',30, 2222)
# msg = redis_client.get('ooo')
# print(msg)

# Models
mistral = LLM("mistral-7b-openorca.Q4_0.gguf","### Human:\n\n### Assistant:\n")
orca2 = LLM("orca-2-7b.Q4_0.gguf","### Human:\n\n### Assistant:\n")
falcon = LLM("gpt4all-falcon-q4_0.gguf","### Instruction:\n\n### Response:\n")
competing_models = [mistral, orca2]
judging_model = falcon

judging_model_prompt = judging_model_prompt = "Given the following question '{}', don't answer it. Instead, give a ranking on a " \
                       "scale of 0 - 1.0 of how similar the following two answers are to one another:\n1. {}\n2. {}\nRESPOND JUST WITH YOUR RATING NUMBER!"
      


def read_questions_from_csv(file_path):
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        questions = [row['Question'] for row in csv_reader]
    return questions
           
def wolfram_alpha_short_answer_query(query, app_id):
    cached_answer = redis_client.get(query)
    if cached_answer:
        return cached_answer
    else:
        base_url = "https://api.wolframalpha.com/v1/result"
        encoded_query = "+".join(query.split())  # URL encode the query
        url = f"{base_url}?i={encoded_query}&appid={app_id}"
        response = requests.get(url)

    if response.status_code == 200:
        answer = response.text.strip()
        redis_client.set(query, answer)
        redis_client.expire(query, KEY_TTL)
        return answer
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None
    
def extract_first_float(s):
    words = s.split()

    for word in words:
        try:
            float_value = float(word)
            return float_value
        except ValueError:
            pass  # Ignore words that can't be converted to float    
    return None    

def main():         
    questions = read_questions_from_csv('General_Knowledge_Questions.csv')[-7:]
    wolfram_answered_questions = []
    results = [] 
    
    for question in questions:
        wolfram_answer = wolfram_alpha_short_answer_query(question, WOLFRAM_APP_ID)
        print(wolfram_answer) 
        if wolfram_answer is not None:
            wolfram_answered_questions.append(question)
    for question in wolfram_answered_questions:
        wolfram_answer = wolfram_alpha_short_answer_query(question, WOLFRAM_APP_ID)
        print(wolfram_answer) 
        if wolfram_answer is not None:        
            model_engine = None
            current_question_results = []
            for model in competing_models:
                model_engine = GPT4All(model.name, MODELS_PATH) 
                full_prompt = model.create_full_prompt(question + " ANSWER WITH AT MOST 5 WORDS")
                print(question)
                start = time.time()
                answer = model_engine.generate(full_prompt, max_tokens=50, temp=0.0)
                end = time.time()
                current_question_results.append({
                    'Question' : question,
                    'Model' : model.name,
                    'Answer' : answer,
                    'TimeInMillisecondsToGetAnswer' : int((end - start) * 1000),
                    'Correctness' : None
                })
                # Here I want to append to 'results' the all the columns but 'Correctness'
                print(f"{model.name} answered:\n{answer}")       
            model_engine = GPT4All(judging_model.name, MODELS_PATH)
            for result in current_question_results:
                prompt = judging_model_prompt.format(question, wolfram_answer, result['Answer'])
                full_prompt = model.create_full_prompt(prompt)
                response = model_engine.generate(full_prompt, max_tokens=20)
                print(response) # Check
                rating = extract_first_float(response)
                if rating is None:
                    raise ValueError
                result['Correctness'] = rating
            results.extend(current_question_results)
            
    print(results)
if __name__ == '__main__':
    main()         
import os
import requests
import csv 
import redis
import time
from llm import LLM
from gpt4all import GPT4All
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file
WOLFRAM_APP_ID = os.getenv("WOLFRAMPLHA_APP_ID")
MODELS_PATH = os.getenv("MODELS_PATH")

KEY_TTL = 14400 # Redis key time-to-live is 4 hours
redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)

# Models
mistral_oo = LLM("mistral-7b-openorca.Q4_0.gguf", "### Human:\n\n### Assistant:\n")
orca2 = LLM("orca-2-7b.Q4_0.gguf", "### Human:\n\n### Assistant:\n")
falcon = LLM("gpt4all-falcon-q4_0.gguf", "### Instruction:\n\n### Response:\n")
competing_models = [falcon, orca2]
judging_model = mistral_oo

def get_assesment_prompt(question, worlfram_answer, model_answer):
    judging_model_prompt = f"""Consider the question: {question}. Please avoid solving it. Instead, assess the similarity between two given 
answers on a scale from 0 to 1.0. 
The provided answers are:
    1. {worlfram_answer}
    2. {model_answer}
Output a similarity score indicating the degree of likeness between the two responses. RETURN ONLY THE SCORE AND NOTHING ELSE."""
     
    return judging_model_prompt
    
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
 
def get_statistics(models, results):   
    statistics_dict = {model.name :
        {'Rating sum': 0, 'Lowest rating' : 1000, 'Lowest rated question' : "", 'Lowest rated answer' : "" } 
                       for model in models}
    for result in results:
      model_name = result['Model']
      current_rating = result['Correctness']     
      statistics_dict[model_name]['Rating sum'] += current_rating
      if current_rating < statistics_dict[model_name]['Lowest rating']:
          statistics_dict[model_name]['Lowest rating'] = current_rating
          statistics_dict[model_name]['Lowest rated question'] = result['Question']
          statistics_dict[model_name]['Lowest rated answer'] = result['Answer']
    return statistics_dict

# This function is for protection only,
# as the response of the judging model is almost always only a float and nothing else.
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
    questions = read_questions_from_csv('General_Knowledge_Questions.csv')
    wolfram_answered_questions = []
    num_answered = 0
    results = [] 
    
    for question in questions:
        print(f'{question}')
        wolfram_answer = wolfram_alpha_short_answer_query(question, WOLFRAM_APP_ID)
        if wolfram_answer is not None:
            print(f'Wolfram answered: {wolfram_answer}\n')
            wolfram_answered_questions.append(question)
        else:
            print('Skipping to next question.\n')
    num_answered = len(wolfram_answered_questions)
    print(num_answered)         
    model_engine = None                    
    for model in competing_models:
        print('Loading LLM...')
        model_engine = GPT4All(model.name, MODELS_PATH)
        print(f'===================== {model.name} =====================\n') 
        for question in wolfram_answered_questions:
            full_prompt = model.create_full_prompt(question + " ANSWER WITH AT MOST 5 WORDS")
            print(question)
            start = time.time()
            answer = model_engine.generate(full_prompt, max_tokens=50, temp=0.0)
            end = time.time()
            results.append({
                'Question' : question,
                'Model' : model.name,
                'Answer' : answer,
                'TimeInMillisecondsToGetAnswer' : int((end - start) * 1000),
                'Correctness' : None
            })
            print(f"{model.name} answered:\n{answer}\n") 
    print('===================== Rating Stage =====================\n')
    print(f'Judging model: {judging_model.name}\n') 
    model_engine = GPT4All(judging_model.name, MODELS_PATH)
    for result in results:
        wolfram_answer = wolfram_alpha_short_answer_query(result['Question'], WOLFRAM_APP_ID)
        prompt = get_assesment_prompt(question, wolfram_answer, result['Answer'])
        full_prompt = model.create_full_prompt(prompt)
        print('Rating answer...')
        response = model_engine.generate(full_prompt, max_tokens=30, temp=0.0) 
        print(response)
        rating = extract_first_float(response)
        if rating is None:
            rating = 0.5 # Give 0.5 for answers who couldn't be rated by the judge LLM
            print(f'{judging_model.name} failed to give a rating (0.5 default rating was given).')
        else:  
            print(f'Answer rated: {judging_model.name} gave a rating of {rating}\n')
        result['Correctness'] = rating 
    print('\nAssesment over. getting statistics ...\n')       
    stats_dict = get_statistics(competing_models, results)
    print(f"""===========Statistics===========\n 
1. Number of questions Wolfram answered: {num_answered}\n
2. Average answer rating of {competing_models[0].name}: {stats_dict[competing_models[0].name]['Rating sum'] / num_answered}\n
3. Average answer rating of {competing_models[1].name}: {stats_dict[competing_models[1].name]['Rating sum'] / num_answered}\n
4. Lowest rating question and answer of {competing_models[0].name}:
    Q - {stats_dict[competing_models[0].name]['Lowest rated question']}
    A - {stats_dict[competing_models[0].name]['Lowest rated answer']}\n
5. Lowest rating question and answer of {competing_models[1].name}:
    Q - {stats_dict[competing_models[1].name]['Lowest rated question']}
    A - {stats_dict[competing_models[1].name]['Lowest rated answer']}\
""")
if __name__ == '__main__':
  main()
  

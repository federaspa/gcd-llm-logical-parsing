import openai
import backoff
import os
import json
from typing import Any
import time
# from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import HumanMessage, SystemMessage

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def make_batch_input_file(
    messages_dict: dict[list[dict[str,Any]]],
    model: str,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_dict: Dict of messages to be sent to OpenAI API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        stop_words: List of words to stop the model from generating.
    Returns:
        List of responses from OpenAI API.
    """
    
    if not os.path.exists('./.tmp'):
        os.makedirs('./.tmp')
    
    for message in messages_dict:
        # append the task description to a jsonl file
        with open('./.tmp/batch_requests.jsonl', 'a') as f:
            f.write(
                json.dumps(
                    {
                        "custom_id": str(message['custom_id']),
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": message['messages'],
                            "temperature": temperature,
                            "top_p": top_p
                        }
                    }
                )
            )
            f.write('\n') 
            
    return './.tmp/batch_requests.jsonl'     

class DispatchOpenAIRequests:
    def __init__(self, API_KEY, model_name, dataset_name, client
                #  stop_words, max_new_tokens
                 ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name   
        self.client = client 
        # self.max_new_tokens = max_new_tokens
        # self.stop_words = stop_words
    
    def upload_batch_file(self, batch_file_path):
        batch_input_file = self.client.files.create(
            file=open(batch_file_path, "rb"),
            purpose="batch"
        ) 
        
        os.remove(batch_file_path)
        
        return batch_input_file.id
    
    
    def create_batch(self, batch_input_file_id):
        
        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": f"{self.model_name}-{self.dataset_name} batch",
            }
        )
        
        return batch
    
    def batch_chat_generate(self, messages_dict, task_description, temperature = 0.0):
        open_ai_messages_dict = []
        for id, message in messages_dict.items():
            open_ai_messages_dict.append(
                {'custom_id': id,
                "messages":
                    [{"role": "system", "content": task_description},
                    {"role": "user", "content": message}]
                }
            )
            
        batch_input_file = make_batch_input_file(open_ai_messages_dict, self.model_name, temperature, 1.0)    
        
        batch_input_file_id = self.upload_batch_file(batch_input_file)
        
        batch_id = self.create_batch(batch_input_file_id)
        
        return batch_id
        

class OpenAIModel:
    def __init__(self, API_KEY, model_name, dataset_name,
                #  stop_words, max_new_tokens
                 ) -> None:
        
        openai.api_key = API_KEY
        self.model_name = model_name
        self.dataset_name = dataset_name
        
        self.client = openai.Client()
        
        self.dispatch = DispatchOpenAIRequests(API_KEY, model_name, dataset_name, self.client)
        
    def batch_generate(self, messages_dict, task_description, temperature = 0.0):
        if self.model_name in ['gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4o']:
            return self.dispatch.batch_chat_generate(messages_dict, task_description, temperature)
        else:
            raise Exception("Model name not recognized")
        
    def get_batch(self, batch_id):

        return self.client.batches.retrieve(batch_id)
    
    def get_batch_results(self, batch_id):
        
        while True:
            batch = self.get_batch(batch_id)
            
            if batch.status == 'completed':
                
                output_file_id = batch.output_file_id
                
                return self.client.files.content(output_file_id)
            
            elif batch.status == 'failed':
                raise Exception("Batch failed")
            
            time.sleep(5)
            
            
            
    # used for chat-gpt and gpt-4
    def chat_generate(self, input_string, task_description, temperature = 0.0):
        response = chat_completions_with_backoff(
                model = self.model_name,
                messages=[
                        {"role": "system", "content": task_description},
                        {"role": "user", "content": input_string}
                    ],
                response_format={ "type": "json_object" },
                # max_tokens = self.max_new_tokens,
                temperature = temperature,
                top_p = 1.0,
                # stop = self.stop_words
        )
        generated_text = response['choices'][0]['message']['content'].strip()
        return generated_text

    def generate(self, input_string, task_description, temperature = 0.0):
        if self.model_name in ['gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4o']:
            return self.chat_generate(input_string, task_description, temperature)
        else:
            raise Exception("Model name not recognized")
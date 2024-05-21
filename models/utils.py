import backoff  # for exponential backoff
import openai
import os
import asyncio
from typing import Any
from langchain_community.llms import LlamaCpp
from llama_cpp.llama import LlamaGrammar
from llama_cpp import Llama
# from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import HumanMessage, SystemMessage


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

async def dispatch_openai_chat_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        stop_words: List of words to stop the model from generating.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop = stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens) -> None:
        openai.api_key = API_KEY
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words

    # used for chat-gpt and gpt-4
    def chat_generate(self, input_string, task_description, temperature = 0.0):
        response = chat_completions_with_backoff(
                model = self.model_name,
                messages=[
                        {"role": "system", "content": task_description},
                        {"role": "user", "content": input_string}
                    ],
                max_tokens = self.max_new_tokens,
                temperature = temperature,
                top_p = 1.0,
                stop = self.stop_words
        )
        generated_text = response['choices'][0]['message']['content'].strip()
        return generated_text

    def generate(self, input_string, task_description, temperature = 0.0):
        if self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.chat_generate(input_string, task_description, temperature)
        else:
            raise Exception("Model name not recognized")
    
    def batch_chat_generate(self, messages_list, task_description, temperature = 0.0):
        open_ai_messages_list = []
        for message in messages_list:
            open_ai_messages_list.append(
                [
                {"role": "system", "content": task_description},
                 {"role": "user", "content": message}]
            )
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                    open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return [x['choices'][0]['message']['content'].strip() for x in predictions]

    def batch_generate(self, messages_list, task_description, temperature = 0.0):
        if self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.batch_chat_generate(messages_list, task_description, temperature)
        else:
            raise Exception("Model name not recognized")
        
        


class GrammarConstrainedModel:
    def __init__(self,  
                model_path = "GCD/llms/nous-hermes-2-solar-10.7b.Q6_K.gguf", 
                n_ctx = 2048,
                n_gpu_layers = -1, 
                n_batch = 512):

        """
        model_path: The path to the model. The default is "GCD/llms/nous-hermes-2-solar-10.7b.Q6_K.gguf".
        grammar_path: The path to the grammar. The default is "GCD/grammars/grammar_unrolled.gbnf".
        n_ctx: The context length of the model. The default is 1024, but you can change it to 512 if you have a smaller model.
        n_gpu_layers: The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
        n_batch: Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        """
        
        # Make sure the model path is correct for your system!
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx = n_ctx,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            verbose = False
        )

    def invoke(self, 
               user, 
               task_description, 
               raw_grammar, 
               max_tokens=50, 
               temperature=0.0,
               frequency_penalty=0.0,
               repeat_penalty=0.0,
               presence_penalty=0.0,
               top_p=0.9,
               top_k=0,
               stop=['------', '###']):

        """
        user: The user input.
        task_description: The task description.
        raw_grammar: The grammar to use for the model.
        max_tokens: The maximum number of tokens to generate.
        echo: Whether to echo the user input.
        stop: The stop words to use for the model.
        """

        grammar = LlamaGrammar.from_string(raw_grammar, verbose=False)

        result = self.llm.create_chat_completion(
        messages=[
            {"role": "system", "content": task_description},
            {"role": "user", "content": user}
            ],
        max_tokens = max_tokens,
        
        frequency_penalty = frequency_penalty,
        repeat_penalty = repeat_penalty,
        presence_penalty = presence_penalty,
        
        temperature=temperature,
        
        top_p=top_p,
        top_k=top_k,
        
        stop = stop,
        grammar = grammar,
        )
        
        return result
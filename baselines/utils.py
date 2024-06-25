import backoff  # for exponential backoff
import openai
import dotenv
import os
from openai import OpenAI, AsyncOpenAI

dotenv.load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=API_KEY)
aclient = AsyncOpenAI(api_key=API_KEY)
import os
import asyncio
from typing import Any

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(**kwargs):
    return client.completions.create(**kwargs)

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

async def dispatch_openai_chat_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    response_format: dict[str,Any],
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
        aclient.chat.completions.create(model=model,
        messages=x,
        temperature=temperature,
        response_format=response_format,
        top_p=top_p,
        stop = stop_words)
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

class OpenAIModel:
    def __init__(self, model_name, stop_words, max_new_tokens) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words

    # used for chat-gpt and gpt-4
    def chat_generate(self, input_string, response_format, temperature = 0.0):
        response = chat_completions_with_backoff(
                model = self.model_name,
                messages=[
                        {"role": "user", "content": input_string}
                    ],
                response_format = response_format,
                temperature = temperature,
                top_p = 1.0,
                stop = self.stop_words
        )
        generated_text = response.choices[0].message.content.strip()
        return generated_text

    def generate(self, input_string, response_format, temperature = 0.0):
        if self.model_name in ['gpt-4o', 'gpt-3.5-turbo', 'gpt-4-turbo']:
            return self.chat_generate(input_string, response_format, temperature)
        else:
            raise Exception("Model name not recognized")

    def batch_chat_generate(self, messages_list, response_format, temperature = 0.0):
        open_ai_messages_list = []
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "user", "content": message}]
            )
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                    open_ai_messages_list, self.model_name, temperature, response_format, 1.0, self.stop_words
            )
        )
        return [x.choices[0].message.content.strip() for x in predictions]

    def batch_generate(self, messages_list,response_format, temperature = 0.0):
        if self.model_name in ['gpt-4o', 'gpt-3.5-turbo', 'gpt-4-turbo']:
            return self.batch_chat_generate(messages_list, response_format, temperature)
        else:
            raise Exception("Model name not recognized")
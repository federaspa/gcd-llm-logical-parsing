from llama_cpp.llama import LlamaGrammar
from llama_cpp import Llama
import warnings
from pushover import Pushover
import dotenv
import os
import logging
import sys
from datetime import datetime

dotenv.load_dotenv()
API_KEY = os.getenv("API_KEY")
USER_KEY = os.getenv("USER_KEY")

def send_notification(message, title):

    po = Pushover(token=API_KEY)
    po.user(USER_KEY)

    msg = po.msg(message=message)

    msg.set("title", title)

    po.send(msg)
    
def get_logger(script_name):
    
    
    current_datetime = datetime.now().strftime("%d%m%Y_%H%M%S")

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create file handler which logs even debug messages
    log_file_name = f"logs/{script_name}_{current_datetime}.log"
    file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class OSModel:
    def __init__(self,
                 model_path,
                 n_gpu_layers=-1,
                 n_batch=512,
                 verbose=False):
        """
        model_path: The path to the model.
        n_gpu_layers: The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
        n_batch: Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        verbose: Whether to print verbose output.
        """

        # Make sure the model path is correct for your system!
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_threads=1 if n_gpu_layers==-1 else None,
            n_ctx=0,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            verbose=verbose
        )
        if "system role not supported" in self.llm.metadata['tokenizer.chat_template'].lower():
            warnings.warn('System role not supported, adapting format', UserWarning)
        
    def _format_messages(self, task_description:str|None, user:str):
        
        if not task_description:
            return [
                {"role": "user", "content": user}
            ]
        
        elif "system role not supported" in self.llm.metadata['tokenizer.chat_template'].lower():
            warnings.warn('System role not supported, adapting format', UserWarning)
            return [
                {"role": "user", "content": f"{task_description}\n\n{user}"}
                ]
        else:
            return [
                {"role": "system", "content": task_description},
                {"role": "user", "content": user}
            ]

    def invoke(self,
               user:str,
               task_description:str|None=None,
               json_format:bool=False,
               raw_grammar:bool=None,
               top_p:float=0.95,
               top_k:float=50,
               min_p:float=0.1,
               **kwargs):
        """
        user: The user input.
        task_description: The task description.
        raw_grammar: The grammar to use for the model.
        json_format: Whether to format the response as JSON.
        top_p, top_k, min_p: Sampling parameters.
        **kwargs: Additional keyword arguments for create_chat_completion.
        """

        if bool(json_format) * bool(raw_grammar):
            warnings.warn("Using json_format and grammar constraints together is unstable", UserWarning)

        grammar = LlamaGrammar.from_string(raw_grammar, verbose=False) if raw_grammar else None
        response_format = {"type": "json_object"} if json_format else None

        messages = self._format_messages(task_description, user)

        response = self.llm.create_chat_completion(
            messages=messages,
            grammar=grammar,
            response_format=response_format,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            **kwargs
        )
        
        if response['choices'][0]["finish_reason"] != 'stop':
            raise Exception(f'Failed to generate response: stopping reason = "{response["choices"][0]["finish_reason"]}"')
        return response
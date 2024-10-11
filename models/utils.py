from llama_cpp.llama import LlamaGrammar
from llama_cpp import Llama
import warnings
from pushover import Pushover
import dotenv
import os
import logging
import sys

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

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create file handler which logs even debug messages
    log_file_name = f"logs/{script_name}.log"
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
                n_gpu_layers = -1,
                n_batch = 512,
                verbose = False):

        """
        model_path: The path to the model.
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
            n_ctx = 12288,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            verbose = verbose
        )

        # input("Press Enter to continue...")

    def invoke(self,
               user,
               task_description,
               json_format = False,
               raw_grammar = None,
               **kwargs):

        """
        user: The user input.
        task_description: The task description.
        raw_grammar: The grammar to use for the model.
        max_tokens: The maximum number of tokens to generate.
        echo: Whether to echo the user input.
        stop: The stop words to use for the model.
        """

        if bool(json_format)*bool(raw_grammar):
            warnings.warn("Using json_format and grammar constraints together is unstable", UserWarning)

        grammar = LlamaGrammar.from_string(raw_grammar, verbose=False) if raw_grammar else None
        response_format = {"type": "json_object"} if json_format else None


        response = self.llm.create_chat_completion(
        messages=[
            {"role": "system", "content": task_description},
            {"role": "user", "content": user},
            # {"role": "user", "content": task_description + "\nHere's some examples:\n------\n" + user}
            ],
        grammar = grammar,
        response_format=response_format,
        **kwargs
        )

        return response
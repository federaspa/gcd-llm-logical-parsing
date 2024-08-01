from llama_cpp.llama import LlamaGrammar
from llama_cpp import Llama

class GrammarConstrainedModel:
    def __init__(self,
                refiner_path,
                n_ctx = 2048,
                n_gpu_layers = -1,
                n_batch = 512):

        """
        model_path: The path to the model.
        grammar_path: The path to the grammar. The default is "GCD/grammars/grammar_unrolled.gbnf".
        n_ctx: The context length of the model. The default is 1024, but you can change it to 512 if you have a smaller model.
        n_gpu_layers: The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
        n_batch: Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        """

        # Make sure the model path is correct for your system!
        self.llm = Llama(
            model_path=refiner_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx = n_ctx,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            verbose = False
        )

        # input("Press Enter to continue...")

    def invoke(self,
               user,
               task_description,
               raw_grammar,
               max_tokens=50,
               temperature=0.01,
               frequency_penalty=0.0,
               repeat_penalty=1.1,
               presence_penalty=0.0,
               top_p=0.9,
               min_p=0.1,
               top_k=20,
               stop=['------', '###']):

        """
        user: The user input.
        task_description: The task description.
        raw_grammar: The grammar to use for the model.
        max_tokens: The maximum number of tokens to generate.
        echo: Whether to echo the user input.
        stop: The stop words to use for the model.
        """

        grammar = LlamaGrammar.from_string(raw_grammar, verbose=False) if raw_grammar else None

        result = self.llm.create_chat_completion(
        messages=[
            {"role": "system", "content": task_description},
            {"role": "user", "content": user},
            # {"role": "user", "content": task_description + "\nHere's some examples:\n------\n" + user}
            ],
        max_tokens = max_tokens,

        frequency_penalty = frequency_penalty,
        repeat_penalty = repeat_penalty,
        presence_penalty = presence_penalty,

        temperature=temperature,

        top_p=top_p,
        min_p=min_p,

        top_k=top_k,

        stop = stop,
        grammar = grammar
        )

        response = result['choices'][0]['message']['content']

        return response
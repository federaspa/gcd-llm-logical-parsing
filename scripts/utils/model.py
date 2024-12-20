from typing import Optional
from llama_cpp import LlamaGrammar, Llama

class LLMModel:
    def __init__(self, config, default_model_config: dict, llama_cpp_config: dict):
        self.config = config
        self.default_model_config = default_model_config
        self.llm = Llama(f16_kv=True, logits_all=True, **llama_cpp_config)
        
    def invoke(self, prompt: str, model_config: Optional[dict] = None, raw_grammar: Optional[str] = None) -> dict:
        grammar = LlamaGrammar.from_string(raw_grammar, verbose=False) if raw_grammar else None
        model_config = model_config or self.default_model_config
        
        return self.llm.create_completion(
            prompt=prompt,
            grammar=grammar,
            logprobs=1,
            **model_config
        )
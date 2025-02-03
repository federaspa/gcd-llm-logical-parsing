from typing import Dict

class TemplateManager:
    DATASET_TYPES = {
        'FOLIO': 'FOL',
        'FOLIOv2': 'FOL',
        'PARARULE': 'FOL',
        'LogicNLI': 'FOL',
        'LogicNLI_symbolic': 'FOL',
        'ProntoQA': 'FOL',
        'ProofWriter': 'FOL',
        'AR-LSAT': 'SAT'
    }

    MODEL_TEMPLATES = {
        'gemma2': 'resources/prompt_templates/gemma.txt',
        'llama3.1': 'resources/prompt_templates/llama.txt',
        'llama3.2': 'resources/prompt_templates/llama.txt',
        'mistral': 'resources/prompt_templates/mistral.txt',
        'ministral': 'resources/prompt_templates/ministral.txt',
        'qwen2.5': 'resources/prompt_templates/qwen.txt',
    }

    def __init__(self, config):
        self.config = config
        self.type = self._get_type()
        self.prompt_template, self.grammar_templates = self._load_templates()

    def _get_type(self) -> str:
        return self.DATASET_TYPES[self.config.dataset_name]

    def _get_model_template(self) -> str:
        model_base = self.config.model_name.split('-')[0]
        
        path = self.MODEL_TEMPLATES[model_base]
        
        with open(path, 'r') as f:
            return f.read()

    def _load_templates(self) -> Dict[str, str]:
        
        model_template = self._get_model_template()
        
        with open(f'./resources/prompts/{self.config.dataset_name}/{self.config.prompt_type}.txt') as f:
            content = f.read()
            prompt_template = model_template.replace('[[user]]', content)

        grammar_template_paths = {
            'json': f'./resources/grammars/json.gbnf',
            'constrained': f'./resources/grammars/{self.type}.gbnf'
        }
       
        grammar_templates = {
            'unconstrained': None
        }
        for key, path in grammar_template_paths.items():
            with open(path, 'r') as f:
                content = f.read()
                grammar_templates[key] = content       

        
        return prompt_template, grammar_templates
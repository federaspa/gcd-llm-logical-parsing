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

    PROMPT_TEMPLATES = {
        'gemma': 'prompts/prompt_templates/gemma.txt',
        'llama': 'prompts/prompt_templates/llama.txt',
        'mistral': 'prompts/prompt_templates/mistral.txt',
        'ministral': 'prompts/prompt_templates/ministral.txt',
        'qwen': 'prompts/prompt_templates/qwen.txt',
        'qwen2.5': 'prompts/prompt_templates/qwen.txt',
    }

    def __init__(self, config):
        self.config = config
        self.type = self._get_type()
        self.templates = self._load_templates()

    def _get_type(self) -> str:
        return self.DATASET_TYPES[self.config.dataset_name]

    def _get_prompt_template(self) -> str:
        model_base = self.config.model_name.split('-')[0]
        return self.PROMPT_TEMPLATES[model_base]

    def _load_templates(self) -> Dict[str, str]:
        template_paths = {
            'unconstrained': f'./prompts/conversion/{self.config.dataset_name}/unconstrained.txt',
            'json': f'./prompts/conversion/{self.config.dataset_name}/json.txt',
            'constrained': f'./prompts/conversion/{self.config.dataset_name}/constrained.txt',
            'constructs': f'./prompts/conversion/{self.config.dataset_name}/constructs.txt',
            'prompt_template': self._get_prompt_template(),
            'json_grammar': f'./LLMs/grammars/json.gbnf',
            'construncts_grammar': f'./LLMs/grammars/{self.type}_constructs.gbnf',
            'constrained_grammar': f'./LLMs/grammars/{self.type}_constrained.gbnf'
        }

        templates = {}
        for key, path in template_paths.items():
            with open(path, 'r') as f:
                content = f.read()
                if key in ['json', 'unconstrained', 'constrained', 'constructs']:
                    with open(template_paths['prompt_template'], 'r') as pt:
                        prompt_template = pt.read()
                        content = prompt_template.replace('[[user]]', content)
                templates[key] = content
        
        return templates
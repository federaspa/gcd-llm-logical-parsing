# from abc import ABC, abstractmethod
# from utils.template_manager import TemplateManager
# from typing import Dict

# class BasePrompter(ABC):
#     def __init__(self, config: dict):
#         self.config = config
#         self.template_manager = TemplateManager(config)

#     @abstractmethod
#     def build_grammar(self) -> str:
#         pass

#     def format_prompt(self, sample: Dict) -> str:
#         problem = '\n'.join(sample['context'])
#         question = sample['question'].strip()
#         return self.template_manager.prompt_template.replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question)
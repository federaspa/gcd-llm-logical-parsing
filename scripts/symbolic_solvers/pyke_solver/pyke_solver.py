import os
import json
from pyke import knowledge_engine
import re

class Pyke_Program:
    def __init__(self, logic_program:str, dataset_name = 'ProntoQA') -> None:
        self.logic_program = logic_program
        self.flag, self.formula_error = self.parse_logic_program(), ""
        self.dataset_name = dataset_name
        
        # create the folder to save the Pyke program
        cache_dir = os.path.join(os.path.dirname(__file__), '.cache_program')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

        # prepare the files for facts and rules
        try:
                self.create_fact_file(self.Facts)
                self.create_rule_file(self.Rules)
                self.flag = True
        except:
            self.flag = False

        self.answer_map = {'ProntoQA': self.answer_map_prontoqa, 
                           'ProofWriter': self.answer_map_proofwriter}

    def parse_logic_program(self):
        try:
            # Parse the JSON string
            program_data = self.logic_program
            
            # Extract the components
            self.Predicates = program_data.get('predicates', [])
            self.Facts = program_data.get('facts', [])
            self.Rules = program_data.get('rules', [])
            
            # Query is stored as a string in JSON but needs to be in a list for compatibility
            query = program_data.get('query', '')
            self.Query = [query] if query else []
            
            return self.validate_program()
        except Exception as e:
            return False

    # check if the program is valid; if not, try to fix it
    def validate_program(self):
        if not self.Rules is None and not self.Facts is None:
            if not self.Rules[0] == '' and not self.Facts[0] == '':
                return True
        # try to fix the program
        tmp_rules = []
        tmp_facts = []
        statements = self.Facts if self.Facts is not None else self.Rules
        if statements is None:
            return False
        
        for fact in statements:
            if fact.find('>>>') >= 0: # this is a rule
                tmp_rules.append(fact)
            else:
                tmp_facts.append(fact)
        self.Rules = tmp_rules
        self.Facts = tmp_facts
        return False
    
    def create_fact_file(self, facts):
        with open(os.path.join(self.cache_dir, 'facts.kfb'), 'w') as f:
            for fact in facts:
                # check for invalid facts
                if not fact.find('$x') >= 0:
                    f.write(fact + '\n')

    def create_rule_file(self, rules):
        pyke_rules = []
        for idx, rule in enumerate(rules):
            pyke_rules.append(self.parse_forward_rule(idx + 1, rule))

        with open(os.path.join(self.cache_dir, 'rules.krb'), 'w') as f:
            f.write('\n\n'.join(pyke_rules))

    # example rule: Furry($x, True) && Quite($x, True) >>> White($x, True)
    def parse_forward_rule(self, f_index, rule):
        premise, conclusion = rule.split('>>>')
        premise = premise.strip()
        # split the premise into multiple facts if needed
        premise = premise.split('&&')
        premise_list = [p.strip() for p in premise]

        conclusion = conclusion.strip()
        # split the conclusion into multiple facts if needed
        conclusion = conclusion.split('&&')
        conclusion_list = [c.strip() for c in conclusion]

        # create the Pyke rule
        pyke_rule = f'''fact{f_index}\n\tforeach'''
        for p in premise_list:
            pyke_rule += f'''\n\t\tfacts.{p}'''
        pyke_rule += f'''\n\tassert'''
        for c in conclusion_list:
            pyke_rule += f'''\n\t\tfacts.{c}'''
        return pyke_rule
    
    '''
    for example: Is Marvin from Mars?
    Query: FromMars(Marvin, $label)
    '''
    def check_specific_predicate(self, subject_name, predicate_name, engine):
        results = []
        with engine.prove_goal(f'facts.{predicate_name}({subject_name}, $label)') as gen:
            for vars, plan in gen:
                results.append(vars['label'])

        with engine.prove_goal(f'rules.{predicate_name}({subject_name}, $label)') as gen:
            for vars, plan in gen:
                results.append(vars['label'])

        if len(results) == 1:
            return results[0]
        elif len(results) == 2:
            return results[0] and results[1]
        elif len(results) == 0:
            return None

    '''
    Input Example: Metallic(Wren, False)
    '''
    def parse_query(self, query):
        pattern = r'(\w+)\(([^,]+),\s*([^)]+)\)'
        match = re.match(pattern, query)
        if match:
            function_name = match.group(1)
            arg1 = match.group(2)
            arg2 = match.group(3)
            arg2 = True if arg2 == 'True' else False
            return function_name, arg1, arg2
        else:
            raise ValueError(f'Invalid query: {query}')

    def execute_program(self):
        # delete the compiled_krb dir
        complied_krb_dir = './scripts/compiled_krb'
        if os.path.exists(complied_krb_dir):
            print('removing compiled_krb')
            os.system(f'rm -rf {complied_krb_dir}/*')

        # absolute_path = os.path.abspath(complied_krb_dir)
        # print(absolute_path)
        try:
            engine = knowledge_engine.engine(self.cache_dir)
            engine.reset()
            engine.activate('rules')
            engine.get_kb('facts')

            # parse the logic query into pyke query
            predicate, subject, value_to_check = self.parse_query(self.Query[0])
            result = self.check_specific_predicate(subject, predicate, engine)
            answer = self.answer_map[self.dataset_name](result, value_to_check)
        except Exception as e:
            return None, str(e)
        
        return answer, ""

    def answer_mapping(self, answer):
        return answer
        
    def answer_map_prontoqa(self, result, value_to_check):
        if result == value_to_check:
            return 'A'
        else:
            return 'B'

    def answer_map_proofwriter(self, result, value_to_check):
        if result is None:
            return 'C'
        elif result == value_to_check:
            return 'A'
        else:
            return 'B'


if __name__ == "__main__":
    # Example JSON input
    json_input = """
    {
      "predicates": [
        "Cold($x, bool)",
        "Rough($x, bool)"
      ],
      "facts": [
        "Quiet(Dave, True)",
        "Red(Dave, True)",
        "Smart(Bob, True)",
        "Kind(Charlie, False)",
        "Cold(Quiet, True)",
        "Cold(Thing, True)"
      ],
      "rules": [
        "Quiet($x, True) && Cold($x, True) && Smart($x, True)>>>Smart(Dave, True)",
        "Red(Charlie, False)&&Rough(Charlie, False)>>>Kind(Charlie, True)",
        "Quiet(Bob, True)&& Rough(Quiet, True)>>>Quiet(Rough, True)",
        "Cold($x, True) && Smart(Quiet, True)>>>Smart(Dave, True)"
      ],
      "query": "Kind(Charlie, False)"
    }
    """
    
    pyke_program = Pyke_Program(json_input, 'ProofWriter')
    result, error_message = pyke_program.execute_program()
    print(f"Result: {result}")
    if error_message:
        print(f"Error: {error_message}")

    complied_krb_dir = './compiled_krb'
    if os.path.exists(complied_krb_dir):
        print('removing compiled_krb')
        os.system(f'rm -rf {complied_krb_dir}')
import json
import os
from nltk.sem.logic import *

os.environ['PROVER9'] = './models/symbolic_solvers/Prover9/bin'

file = 'fixed_errors_folio.json'

known_errors = {}

error_types = {}

with open(file, 'r') as f:
    samples_dict = json.load(f)
    
possible_errors = ["quantifier_misplacement",
"parentheses_manipulation",
"missing_quantifier",
"nested_predicates",
"mixed"]

error_map = {
    'qm': 'quantifier_misplacement',
    'mq': 'missing_quantifier',
    'pm': 'parentheses_manipulation',
    'np': 'nested_predicates',
    'mix': 'mixed',
}


for key in samples_dict.keys():
    samples = samples_dict[key]
    for sample in samples:
        
        error_types[sample['id']] = []
        
        for formula in sample['diff']:
            
            error = formula['raw'].split(':::')[0].strip()
            
            if error in known_errors.keys():
                error_type = known_errors[error]
                error_types[sample['id']].append(error_type)
                continue

            
            print(error)
                
            while True:
                
                error_type = input().strip()
                
                if error_type == 'break':
                    raise KeyboardInterrupt()
                
                elif error_type == 'skip':
                    break
                
                if not error_type in error_map.keys():
                    print('Invalid input.')
                    continue
                
                error_type = error_map[error_type]
                
                break
                
            if error_type == 'skip':
                continue

            
            # if error_type == 'skip':
            #     skipped = True
            #     break

                
            # if error_type == 'skip':
            #     continue
            
            known_errors[error] = error_type     
                   
            error_types[sample['id']].append((error, error_type))
            

        # if skipped:
        #     sample["fixed"] = False

        # else:
    #     sample["fixed"] = True
        
        with open('error_types.json', 'w') as f:
            json.dump(error_types, f, ensure_ascii=False, indent=4)
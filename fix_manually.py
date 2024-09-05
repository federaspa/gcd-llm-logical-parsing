import json
import os
from nltk.sem.logic import *

from models.symbolic_solvers.fol_solver.Formula_util import FOL_Formula

os.environ['PROVER9'] = './models/symbolic_solvers/Prover9/bin'

file = 'evaluation/qualitative/qualitative_nli/4o/fixed_errors.json'

known_errors = {}

with open(file, 'r') as f:
    ers = json.load(f)
    
with open('evaluation/quantitative/wrong_ids.json', 'r') as f:
    wrong_ids = json.load(f)
    
# samples = samples['gcd_llama-2-7b']
for key, samples in ers.items():
    for sample in samples:
                
        if "fixed" in sample.keys():
            if sample["fixed"]:
                continue
            
        if sample['id'] in wrong_ids:
            sample['fixed'] = False
            continue
        
        fixed_errors = []
        sample["manual_prog"] = {
            "First-Order-Logic Rules": sample["raw_prog"]["First-Order-Logic Rules"],
            "First-Order-Logic Question": sample["raw_prog"]["First-Order-Logic Question"]
        }
        
        skipped = False
        
        for formula in sample['diff']:
            
            error = formula['raw'].split(':::')[0].strip()
            nl = formula['raw'].split(':::')[1].strip()
            
            if error in known_errors.keys():
                fixed = known_errors[error]
                fixed_errors.append((error, fixed))
                continue

            
            print(error, nl)
            
            not_valid = True
            
            while not_valid:
                
                fixed = input().strip()
                
                if fixed == 'break':
                    raise KeyboardInterrupt()
                
                if fixed == 'skip':
                    skipped = True
                    break

                fol_formula = FOL_Formula(fixed)

                if fol_formula.is_valid:
                    not_valid = False
                else:
                    print('Not valid')
                
            if fixed == 'skip':
                continue
            
            known_errors[error] = fixed            
            fixed_errors.append((error, fixed))
            
        # try:
            
        if isinstance(sample["raw_prog"]["First-Order-Logic Rules"], list):
            sample["manual_prog"]["First-Order-Logic Rules"] = '\n'.join(sample["manual_prog"]["First-Order-Logic Rules"])
            
        if isinstance(sample["manual_prog"]["First-Order-Logic Question"], list):
            sample["manual_prog"]["First-Order-Logic Question"] = '\n'.join(sample["manual_prog"]["First-Order-Logic Question"])
                
        for error, fixed in fixed_errors:
            
            sample["manual_prog"]["First-Order-Logic Rules"] = sample["manual_prog"]["First-Order-Logic Rules"].replace(error, fixed)
            sample["manual_prog"]["First-Order-Logic Question"] = sample["manual_prog"]["First-Order-Logic Question"].replace(error, fixed)
        # except Exception as e:
        #     sample["fixed"] = False
        #     print(e)
        #     continue

        if skipped:
            sample["fixed"] = False

        else:
            sample["fixed"] = True
            
        with open(file, 'w') as f:
            json.dump(ers, f, ensure_ascii=False, indent=4)
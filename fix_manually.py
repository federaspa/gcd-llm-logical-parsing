import json
import os
from nltk.sem.logic import *

from models.symbolic_solvers.fol_solver.Formula_util import FOL_Formula

os.environ['PROVER9'] = './models/symbolic_solvers/Prover9/bin'


with open('checkpoint.json', 'r') as f:
    samples = json.load(f)
    
# samples = samples['gcd_llama-2-7b']
for sample in samples:
    
    if "fixed" in sample.keys():
        if sample["fixed"]:
            continue
    
    fixed_errors = []
    sample["manual_prog"] = {
        "First-Order-Logic Rules": sample["raw_prog"]["First-Order-Logic Rules"],
        "First-Order-Logic Question": sample["raw_prog"]["First-Order-Logic Question"]
    }
    
    for formula in sample['diff']:
        
        error = formula['raw'].split(':::')[0].strip()
        print(error)
        
        not_valid = True
        
        while not_valid:
            
            fixed = input().strip()
            
            if fixed == 'break':
                raise KeyboardInterrupt()
            
            if fixed == 'skip':
                break

            fol_formula = FOL_Formula(fixed)

            if fol_formula.is_valid:
                not_valid = False
            else:
                print('Not valid')
            
        if fixed == 'skip':
            continue

        fixed_errors.append((error, fixed))
        
    try:
        for error, fixed in fixed_errors:
            sample["manual_prog"]["First-Order-Logic Rules"] = sample["manual_prog"]["First-Order-Logic Rules"].replace(error, fixed)
            sample["manual_prog"]["First-Order-Logic Question"] = sample["manual_prog"]["First-Order-Logic Question"].replace(error, fixed)
    except Exception as e:
        sample["fixed"] = False
        continue

    if fixed == 'skip':
        sample["fixed"] = False

    else:
        sample["fixed"] = True
        
    with open('checkpoint.json', 'w') as f:
        json.dump(samples, f, ensure_ascii=False)
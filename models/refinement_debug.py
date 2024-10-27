from refinement import *

class SelfRefinementEngine(SelfRefinementEngine):
    def single_round_self_refinement(self):
        logic_problems = self._load_logic_problems()
        
        for sample in logic_problems:

            logic_problem:dict = sample.get('logic_problem', {})
            if not logic_problem:
                continue

            _, status, error = self._safe_execute_program(logic_problem)
            
            if status == 'parsing error' and error:
                
                nl = '\n'.join(r for r in sample['nl_problem']['nl_rules'])
                fol = '\n'.join(r for r in logic_problem['fol_rules'])
                
                print(sample['id'])
                print(f"Original NL:\n\n{nl}\n")
                print(f"Original FOL:\n\n{fol}\n")
                print("Error:", error)
                
                input("Press Enter to continue...")
                
                print("Reasoning...")
                reasoning, _ = self._parsing_reasoning_generator(logic_problem, error)
                
                print(reasoning)
                
                print("Fixing...")
                correction, _ = self._parsing_correction_generator(logic_problem, error, reasoning)
                

                print("Correction:", correction)    

if __name__ == "__main__":
    config = parse_args()
    
    refiner = OSModel(
        model_path=config.refiner_path,
        n_gpu_layers=config.n_gpu_layers,
        verbose=config.verbose
    )
    
    try:
        for round in range(config.starting_round, config.maximum_rounds + 1):
            engine = SelfRefinementEngine(config, round, refiner=refiner)
            engine.single_round_self_refinement()
            
    except KeyboardInterrupt:
        sys.exit(0)
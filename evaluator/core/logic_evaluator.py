from utils.metrics import MetricsCalculator

class LogicEvaluator:
    @staticmethod
    def evaluate_sample_groups(samples):
        n = 204
        
        if len(samples) != n:
            return {
            'UNCONSTRAINED': {
            'accuracy': -1,
            'coverage': -1
        },
            'JSON': {
            'accuracy': -1,
            'coverage': -1
        },
            'FOL': {
            'accuracy': -1,
            'coverage': -1
        }}
        
        sample_groups = {
            'unconstrained': [s['logic_problem'] for s in samples if 'logic_problem' in s],
            'json': [s['logic_problem_json'] for s in samples if 'logic_problem_json' in s],
            'fol': [s['logic_problem_gcd'] for s in samples if 'logic_problem_gcd' in s]
        }
        
        metrics = {
            'UNCONSTRAINED': MetricsCalculator.compute_metrics(sample_groups['unconstrained'], n),
            'JSON': MetricsCalculator.compute_metrics(sample_groups['json'], n),
            'FOL': MetricsCalculator.compute_metrics(sample_groups['fol'], n)
        }
        
        return metrics
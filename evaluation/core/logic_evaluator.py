from utils.metrics import MetricsCalculator

class LogicEvaluator:
    @staticmethod
    def evaluate_sample_groups(samples, with_string=False):
        n = len(samples)
        
        sample_groups = {
            'unconstrained': [s['logic_problem'] for s in samples if 'logic_problem' in s],
            'json': [s['logic_problem_json'] for s in samples if 'logic_problem_json' in s],
            'constrained': [s['logic_problem_gcd'] for s in samples if 'logic_problem_gcd' in s]
        }
        
        metrics = {
            'UNCONSTRAINED': MetricsCalculator.compute_metrics(sample_groups['unconstrained'], n, with_string),
            'JSON': MetricsCalculator.compute_metrics(sample_groups['json'], n, with_string),
            'CONSTRAINED': MetricsCalculator.compute_metrics(sample_groups['constrained'], n, with_string)
        }
        
        # baseline = MetricsCalculator.compute_baseline_metrics(samples, with_string)
        # metrics.update({'RANDOM': baseline[0], 'FIXED': baseline[1]})
        
        return metrics
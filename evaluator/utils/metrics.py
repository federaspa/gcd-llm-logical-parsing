import numpy as np
from utils.parsers import AnswerParser

class MetricsCalculator:
    @staticmethod
    def compute_ratio(numerator, denominator):
        
        return numerator/denominator if denominator else 0
         

    @staticmethod
    def compute_metrics(samples, n):
        parsable_samples = [s for s in samples if s['status'] != 'parsing error']
        executable_samples = [s for s in samples if s['status'] == 'success']
        
        gold_answers, predictions = AnswerParser.parse_answers(samples)
        correct_predictions = sum(g == p for g, p in zip(gold_answers, predictions))
        
        metrics = {
            'accuracy': MetricsCalculator.compute_ratio(correct_predictions, n),
            'coverage': MetricsCalculator.compute_ratio(len(executable_samples), n)
        }
        
        return metrics

    @staticmethod
    def compute_baseline_metrics(samples):
        gold_answers = [s['nl_problem']['answer'] for s in samples]
        random_answers = np.random.choice(['A', 'B', 'C', 'N/A'], len(samples))
        
        correct_random = sum(g == p for g, p in zip(gold_answers, random_answers))
        
        return (
            MetricsCalculator.compute_ratio(correct_random, len(samples)),
        )
        
    @staticmethod
    def compute_improvements(metrics, key1, key2):
        
        improvement_accuracy = metrics[key1]['accuracy'] - metrics[key2]['accuracy']
        improvement_coverage = metrics[key1]['coverage'] - metrics[key2]['coverage']
        
        improvement = {
            'accuracy': improvement_accuracy,
            'coverage': improvement_coverage
        }
        
        return improvement
    
    @staticmethod
    def evaluate_sample_groups(samples, n=0):        
        if len(samples) != n:
            print(len(samples), '!=', n)
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
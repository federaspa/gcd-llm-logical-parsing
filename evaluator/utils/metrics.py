import numpy as np
from utils.parsers import AnswerParser

class MetricsCalculator:
    @staticmethod
    def compute_ratio(numerator, denominator, with_string=False):
        ratio = numerator/denominator if denominator else 0
        return (ratio, f'{numerator}/{denominator}') if with_string else ratio

    @staticmethod
    def compute_metrics(samples, n, with_string=False):
        parsable_samples = [s for s in samples if s['status'] != 'parsing error']
        executable_samples = [s for s in samples if s['status'] == 'success']
        
        gold_answers, predictions = AnswerParser.parse_answers(samples)
        correct_predictions = sum(g == p for g, p in zip(gold_answers, predictions))
        
        metrics = {
            'accuracy': MetricsCalculator.compute_ratio(correct_predictions, n, with_string),
            'coverage': MetricsCalculator.compute_ratio(len(executable_samples), n, with_string)
        }
        
        return metrics

    @staticmethod
    def compute_baseline_metrics(samples, with_string=False):
        gold_answers = [s['nl_problem']['answer'] for s in samples]
        random_answers = np.random.choice(['A', 'B', 'C', 'N/A'], len(samples))
        fixed_answers = ['A']*len(samples)
        
        correct_random = sum(g == p for g, p in zip(gold_answers, random_answers))
        correct_fixed = sum(g == p for g, p in zip(gold_answers, fixed_answers))
        
        return (
            MetricsCalculator.compute_ratio(correct_random, len(samples), with_string),
            MetricsCalculator.compute_ratio(correct_fixed, len(samples), with_string)
        )
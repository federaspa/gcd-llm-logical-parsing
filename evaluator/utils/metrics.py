import numpy as np
from utils.parsers import AnswerParser

class MetricsCalculator:
    @staticmethod
    def compute_ratio(numerator, denominator):
        
        return round(numerator/denominator, 2) if denominator else 0
         

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

    # @staticmethod
    # def compute_baseline_metrics(samples):
    #     gold_answers = [s['nl_problem']['answer'] for s in samples]
    #     random_answers = np.random.choice(['A', 'B', 'C', 'N/A'], len(samples))
    #     fixed_answers = ['A']*len(samples)
        
    #     correct_random = sum(g == p for g, p in zip(gold_answers, random_answers))
    #     correct_fixed = sum(g == p for g, p in zip(gold_answers, fixed_answers))
        
    #     return (
    #         MetricsCalculator.compute_ratio(correct_random, len(samples), with_string),
    #         MetricsCalculator.compute_ratio(correct_fixed, len(samples), with_string)
    #     )
        
    # @staticmethod
    # def compute_improvements(metrics, key1, key2):
        
    #     improvement_accuracy = metrics[key1]['accuracy'] - metrics[key2]['accuracy']
    #     improvement_coverage = metrics[key1]['coverage'] - metrics[key2]['coverage']
        
    #     improvement = {
    #         'accuracy': improvement_accuracy,
    #         'coverage': improvement_coverage
    #     }
        
    #     return improvement
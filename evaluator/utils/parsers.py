import re

class AnswerParser:
    CHOICES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'N/A']
    CHOICE_PATTERNS = [f"{c}{s}" for c in CHOICES for s in ['', ')', '.']]
    ANSWER_INDICATORS = [
        'the correct option is', 'the correct answer is',
        'The correct answer is', 'The correct option is',
        'Thus, the answer is'
    ]

    @staticmethod
    def get_choice(answer_str):
        for pattern in AnswerParser.CHOICE_PATTERNS:
            if answer_str.startswith(pattern):
                return pattern.replace(')', '').replace('.', '')
        
        for indicator in AnswerParser.ANSWER_INDICATORS:
            if indicator in answer_str:
                answer_str = answer_str.split(indicator)[1].strip()
                for pattern in AnswerParser.CHOICE_PATTERNS:
                    if answer_str.startswith(pattern):
                        return pattern.replace(')', '').replace('.', '')
        return None

    @staticmethod
    def parse_answers(samples):
        gold_answers = []
        predictions = []
        
        for sample in samples:
            gold_answer = sample['answer']
            prediction = AnswerParser.get_choice(sample['predicted_answer'].strip())
            gold_answers.append(gold_answer)
            predictions.append(prediction)
            
        return gold_answers, predictions
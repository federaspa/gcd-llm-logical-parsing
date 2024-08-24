import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from sklearn.metrics import f1_score, multilabel_confusion_matrix, confusion_matrix

def safe_execute_program(logic_program):
    program_executor = FOL_Prover9_Program
    program = program_executor(logic_program, 'FOLIO', 'dynamic')
    # cannot parse the program
    if program.flag == False:
        answer = 'N/A'
        return answer, 'parsing error', program.formula_error, program.nl_error
    # execuate the program
    answer, error_message = program.execute_program()
    # not executable
    if answer is None:
        answer =  'N/A'
        return answer, 'execution error', error_message, 'UNK'
    # successfully executed
    answer = program.answer_mapping(answer)
    return answer, 'success', None, None

def main():
    with open('qualitative/llama-2-7b.json', 'r') as f:
        samples = json.load(f)
        
    y_true = []
    y_pred_manual = []
    y_pred_grammar = []
        
    for sample in samples:
        
        if not sample['fixed']:
            continue
        
        
        manual_prog = sample["manual_prog"]
        pred_answer, status, error, _ = safe_execute_program(manual_prog)
        
        y_true.append(sample['answer'])
        y_pred_grammar.append(sample["grammar_answer"])
        y_pred_manual.append(pred_answer)
        
        # if sample['answer'] != pred_answer:
        #     print(sample['id'], sample['answer'], pred_answer)

    # print()        
    # print(f1_score(gold, pred, average=None, labels=['A', 'B', 'C', 'N/A']))
    # print(multilabel_confusion_matrix(y_true, y_pred, labels=['A', 'B', 'C', 'N/A']))
    # print(pd.Series(gold).value_counts())
    # print(pd.Series(pred).value_counts())
    
    labels=['A', 'B', 'C', 'N/A']
    
    data = confusion_matrix(y_true, y_pred_manual)
    df_cm = pd.DataFrame(data, columns=labels, index = labels)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'


    f, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    palette = sns.color_palette("Set2", n_colors=2)

    sns.heatmap(df_cm, cbar=False, annot=True, cmap=cmap, square=True, fmt='.0f',
                annot_kws={'size': 10})
    plt.title('Actuals vs Predicted Heatmap')
    plt.savefig('qualitative/confusion_refine.png')
    
    
    # Create a new plot for y_true and y_pred countplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a dataframe with y_true and y_pred
    df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred_manual})

    # Melt the dataframe to create a long format
    df_melted = pd.melt(df, var_name='category', value_name='label')

    # Create the countplot
    sns.countplot(x='label', hue='category', data=df_melted, ax=ax, order=labels, palette=palette, stat='proportion')

    ax.set_xlabel('Labels')
    ax.set_ylabel('Frequency')
    ax.set_title('Actual vs Predicted Label Distribution')
    ax.legend()

    plt.savefig('qualitative/distribution_refine.png')








        
    
                
if __name__ == '__main__':
    main()
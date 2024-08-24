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
        
    with open('qualitative/gpt-3.5.json', 'r') as f:
        samples_sket = json.load(f)
        
    y_true = []
    y_pred_manual = []
    y_pred_grammar = []
    y_sket = []
        
    for sample in zip(samples):
        
        if not sample['fixed']:
            continue
        
        
        manual_prog = sample["manual_prog"]
        pred_answer, status, error, _ = safe_execute_program(manual_prog)
        
        if pred_answer == 'N/A':
            continue
        
        y_true.append(sample['answer'])
        y_pred_grammar.append(sample["grammar_answer"])
        y_pred_manual.append(pred_answer)
        
    for sample in samples_sket:
        y_sket.append(sample['grammar_answer'])
        
        
        # if sample['answer'] != pred_answer:
        #     print(sample['id'], sample['answer'], pred_answer)

    # print()        
    # print(f1_score(gold, pred, average=None, labels=['A', 'B', 'C', 'N/A']))
    # print(multilabel_confusion_matrix(y_true, y_pred, labels=['A', 'B', 'C', 'N/A']))
    # print(pd.Series(gold).value_counts())
    # print(pd.Series(pred).value_counts())
    
    labels=['A', 'B', 'C']
    
    data_manual = confusion_matrix(y_true, y_pred_manual)
    df_cm_manual = pd.DataFrame(data_manual, columns=labels, index = labels)
    df_cm_manual.index.name = 'Actual'
    df_cm_manual.columns.name = 'Predicted'

    data_grammar = confusion_matrix(y_true, y_pred_grammar)
    df_cm_grammar = pd.DataFrame(data_grammar, columns=labels, index = labels)
    df_cm_grammar.index.name = 'Actual'
    df_cm_grammar.columns.name = 'Predicted'

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    palette = sns.color_palette("Set2", n_colors=3)

    # Plot 1: y_manual
    sns.heatmap(df_cm_manual, cbar=False, annot=True, cmap=cmap, square=True, fmt='.0f',
                annot_kws={'size': 10}, ax=ax1)
    ax1.set_title('Actuals vs Predicted (Manual) Heatmap')

    # Plot 2: y_grammar
    sns.heatmap(df_cm_grammar, cbar=False, annot=True, cmap=cmap, square=True, fmt='.0f',
                annot_kws={'size': 10}, ax=ax2)
    ax2.set_title(u'Actuals vs Predicted (\u2605) Heatmap')

    plt.savefig('qualitative/confusion_refine.png')

    # Create a new plot for y_true and y_pred countplots
    fig, ax1 = plt.subplots(figsize=(18, 6))

    # Create a dataframe with y_true and y_pred
    df = pd.DataFrame({'Actual': y_true, 'Predicted (Manual)': y_pred_manual, u'Predicted (\u2605)': y_pred_grammar})
    # df_grammar = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred_grammar})

    # Melt the dataframe to create a long format
    df_melted = pd.melt(df, var_name='category', value_name='label')
    # df_melted_grammar = pd.melt(df_grammar, var_name='category', value_name='label')

    # Create the countplot
    sns.countplot(x='label', hue='category', data=df_melted, ax=ax1, order=labels, palette=palette)
    ax1.set_xlabel('Labels')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Actual vs Predicted Label Distribution')

    # sns.countplot(x='label', hue='category', data=df_melted_grammar, ax=ax2, order=labels, palette=palette, stat='proportion')
    # ax2.set_xlabel('Labels')
    # ax2.set_ylabel('Frequency')
    # ax2.set_title(u'Actual vs Predicted Label Distribution (\u2605)')

    plt.savefig('qualitative/distribution_refine.png')    
                
if __name__ == '__main__':
    main()
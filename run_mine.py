import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from models.symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from sklearn.metrics import f1_score, multilabel_confusion_matrix, confusion_matrix


label_map = {
    'A': 'True',
    'B': 'False',
    'C': 'Uncertain',
    'N/A': 'N/A'
}

dataset = 'FOLIO'
load_dir = 'qualitative'
load_dir = load_dir + '_' + 'folio' if dataset == 'FOLIO' else load_dir + '_' + 'nli'


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
    with open(f'{load_dir}/llama-2-7b.json', 'r') as f:
        samples = json.load(f)
        
    with open(f'{load_dir}/gpt-3.5.json', 'r') as f:
        samples_sket = json.load(f)
        
    y_true = []
    y_pred_manual = []
    y_pred_grammar = []
    y_pred_sket = []
        
    for sample in tqdm(samples):
        
        if not sample['fixed']:
            continue
        
        if "manual_answer" in sample.keys():
            pred_answer = sample["manual_answer"]
        
        else:
            manual_prog = sample["manual_prog"]
            pred_answer, status, error, _ = safe_execute_program(manual_prog)
            sample["manual_answer"] = pred_answer        

        if pred_answer == 'N/A':
            continue
        
        y_true.append(label_map[sample['answer']])
        y_pred_grammar.append(label_map[sample["grammar_answer"]])
        y_pred_manual.append(label_map[pred_answer])
        
        sket_candidates = [s for s in samples_sket if s['id'] == sample['id']]
        
        if sket_candidates:
            sample_sket = sket_candidates[0]
            y_pred_sket.append(label_map[sample_sket['grammar_answer']])
        else:
            y_pred_sket.append('N/A')
            
    with open(f'{load_dir}/llama-2-7b.json', 'w') as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)
        
        
    # for sample in samples_sket:
    #     y_true_sket.append(label_map[sample['answer']])
    #     y_pred_sket.append(label_map[sample['grammar_answer']])


    labels=['True', 'False', 'Uncertain']
    
    data_manual = confusion_matrix(y_true, y_pred_manual, labels=labels)
    df_cm_manual = pd.DataFrame(data_manual, columns=labels, index = labels)
    df_cm_manual.index.name = 'Actual'
    df_cm_manual.columns.name = 'Predicted'

    data_grammar = confusion_matrix(y_true, y_pred_grammar, labels=labels)
    df_cm_grammar = pd.DataFrame(data_grammar, columns=labels, index = labels)
    df_cm_grammar.index.name = 'Actual'
    df_cm_grammar.columns.name = 'Predicted'
    
    data_sket = confusion_matrix(y_true, y_pred_sket, labels=labels)
    df_cm_sket = pd.DataFrame(data_sket, columns=labels, index = labels)
    df_cm_sket.index.name = 'Actual'
    df_cm_sket.columns.name = 'Predicted'
    
    # increase text size with plt.rcParams
    plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})
    
    dfs = [df_cm_manual, df_cm_grammar, 
        #    df_cm_sket
           ]
    names = ['Manual', '\u2605', 
            #  'GPT-3.5'
             ]

    fig, axes = plt.subplots(1, len(dfs), figsize=(20, 12))
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    palette = sns.color_palette("Set2", n_colors=4)
    annot_kws={'size': 18}

    for ax, df, strat in zip(axes, dfs, names):
        
        sns.heatmap(df, cbar=False, annot=True, cmap=cmap, square=True, fmt='.0f',
                    annot_kws=annot_kws, ax=ax)
        ax.set_title(u'Actual vs Predicted Labels ({strat})\n{dataset}'.format(strat=strat, dataset=dataset))
    
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
        # slightly increase the very tight bounds:
        xpad = 0.05 * width
        ypad = 0.05 * height
        fig.add_artist(plt.Rectangle((x0-xpad, y0-ypad), width+2*xpad, height+2*ypad, edgecolor='black', linewidth=1, fill=False))


    plt.savefig(f'{load_dir}/3.5/confusion_refine_0.png')

    df = pd.DataFrame({
        'Actual': y_true, 
        'Predicted (Manual)': y_pred_manual, 
        u'Predicted (\u2605)': y_pred_grammar, 
        # 'Predicted (GPT-3.5)': y_pred_sket
        })
    df_melted = pd.melt(df, var_name='Strategy', value_name='label')


    fig, ax1 = plt.subplots(figsize=(10, 8))
    sns.countplot(x='label', hue='Strategy', data=df_melted, ax=ax1, order=labels, palette=palette)
    ax1.set_xlabel('Labels')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.set_title(f'Actual vs Predicted Labels \n{dataset}', fontdict={'fontsize': 20})
    
    plt.savefig(f'{load_dir}/3.5/distribution_refine_0.png')    
                
if __name__ == '__main__':
    main()
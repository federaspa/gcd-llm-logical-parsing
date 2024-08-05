
import pandas as pd
import argparse

def change_config(x:str) -> str:
    if x == 'GPT-3.5-Turbo + GPT-3.5-Turbo':
        return 'GGLLM (GPT-3.5-Turbo + GPT-3.5-Turbo)'
    
    elif x == 'GPT-4o + GPT-4o':
        return 'GGLLM (GPT-4o + GPT-4o)'

    else:
        return x
    
def rename_model(x:str) -> str:
    if x == 'gpt-3.5-turbo':
        return 'GPT-3.5-Turbo' 
    
    elif x == 'gpt-4o':
        return 'GPT-4o'

    else:
        return x.replace('-', ' ').replace('b', 'B').capitalize()
    
def rename_strategy(x:str) -> str:
    if 'CoT' in x:
        return 'Chain-of-Thought'

    elif 'LogicLM' in x:
        return 'LogicLM'
    
    else:
        return 'GGLLM'

def make_df(df_path: str, sketcher:str, backup: bool, full: bool = False) -> pd.DataFrame:
    df_original = pd.read_csv(df_path)
    group_cols = ['sketcher', 'refiner', 'gcd', 'finetune', 'baseline', 'use_backup']

    # Define the metrics to average
    metric_cols = ['precision', 'recall', 'f1', 'executable_precision', 'executable_recall', 
                'executable_f1', 'executable_rate', 'parsing_errors_rate', 'execution_errors_rate']

    # Group by the experiment columns and calculate the mean of the metrics
    df = df_original.groupby(group_cols)[metric_cols].mean().reset_index().rename(columns={'gcd': 'GCD'}).rename(columns={'finetune': 'Finetune'})
        
    df = df[df['use_backup'] == backup].drop(columns=['use_backup'])
    df = df[df['sketcher'] == sketcher].drop(columns=['sketcher'])
        
    # df['sketcher'] = df['sketcher'].apply(lambda x: change_cell(x))
    df['refiner'] = df['refiner'].apply(lambda x: rename_model(x))
    df['refiner'] = df['refiner'] + df['Finetune'].map({True: ' (fine-tuned)', False: ''})
    df['strategy'] = df['baseline'].apply(lambda x: rename_strategy(x))

    # df['config'] = df['sketcher'] + ' + ' + df['refiner'] + df['Finetune'].map({True: '(fine-tuned)', False: ''})

    df['baseline'] = df['baseline'].astype(str)
    # df.loc[df['baseline'] != 'False', 'config'] = df.loc[df['baseline'] != False, 'baseline']
    if full:
    # df['config'] = df['config'].apply(lambda x: change_config(x))
        return df.reset_index(drop=True)
    
    df = df[['strategy', 'refiner', 'GCD', 'f1']].sort_values(by='f1', ascending=False).reset_index(drop=True)
    return df
    
def parse_args():
    parser = argparse.ArgumentParser(description='Make a dataframe from the results csv file.')
    parser.add_argument('--df_path', type=str, help='Path to the results csv file.', required=True)
    parser.add_argument('--save_path', type=str, help='Path to save the dataframe.', required=True)
    parser.add_argument('--sketcher', type=str, default='gpt-3.5-turbo', help='The sketcher model to filter the results.')
    parser.add_argument('--backup', action='store_true', help='Whether to use the models with backup models or not.')
    parser.add_argument('--full', action='store_true', help='Whether to return the full dataframe or not.')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    df = make_df(args.df_path, args.sketcher, args.backup, args.full)
    df.to_csv(args.save_path, index=False)
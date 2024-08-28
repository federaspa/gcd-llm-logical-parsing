
import pandas as pd
import numpy as np
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
        return x.replace('-', ' ').capitalize()
    
def rename_strategy(x:str) -> str:
    if 'CoT' in x:
        return 'CoT'

    elif 'LogicLM' in x:
        return 'LogicLM'
    
    else:
        return 'GCLLM'

def make_df(df_path: str) -> pd.DataFrame:
    df_original = pd.read_csv(df_path)
    group_cols = ['sketcher', 'prompt_mode']

    # Define the metrics to average
    metric_cols = ['f1', 'executable_f1','executable_rate']

    # Group by the experiment columns and calculate the mean of the metrics
    df = df_original.groupby(group_cols).agg({col: ['mean', 'std'] for col in metric_cols}).reset_index()
    # make a new column new_f1 with the mean +- the std

    for col in ['f1', 'executable_f1']:
        df[col] = '$' + df[col, 'mean'].apply(lambda x: f'{np.round(x, 2)}') + '^{' + '\\pm ' + df[col, 'std'].apply(lambda y: f'{np.round(y, 2)}') + '}$'
        df.drop(columns=(col, 'std'), inplace=True)

    col = 'executable_rate'
    df[col] = '$' + df[col, 'mean'].apply(lambda x: f'{np.round(x, 4)*100}' if not np.isnan(x) else 'N/A') + '^{' + '\\pm' + df[col, 'std'].apply(lambda y: f'{int(y*100)}' if not np.isnan(y) else '') + '}' + '\\%$'
    df.drop(columns=(col, 'std'), inplace=True)
        
        
    df.columns = df.columns.droplevel(1)
    
    # df = df.groupby(['sketcher']).agg({'f1': 'first', 
    #                                 'f1_nobackup': 'first',
    #                                 'executable_f1_backup': 'first',
    #                                 'executable_f1_nobackup': 'first',
    #                                 'executable_rate_backup': 'first',
    #                                 'backup_f1_executable_backup': 'first',
    #                                 'backup_f1_executable_nobackup': 'first',
    #                                 'backup_f1_non_executable_backup': 'first',
    #                                 'backup_f1_non_executable_nobackup': 'first'
    #                                 }).reset_index()

    
    df['sketcher'] = df['sketcher'].apply(lambda x: rename_model(x))

    df = df.rename(columns={
                            'executable_f1': 'Weighted F1\n(executable samples)',
                            'f1': 'Weighted F1\n(all samples)',
                            'executable_rate': 'Executable Samples (\\%)',
                            'sketcher': 'Sketcher',
                            }) 

    
    return df
    
def parse_args():
    parser = argparse.ArgumentParser(description='Make a dataframe from the results csv file.')
    parser.add_argument('--folio_df_path', type=str, help='Path to the results csv file for FOLIO.', required=True)
    parser.add_argument('--nli_df_path', type=str, help='Path to the results csv file for LogicNLI.', required=True)
    # parser.add_argument('--save_path', type=str, help='Path to save the dataframe.', required=True)
    # parser.add_argument('--save_path_backup', type=str, help='Path to save the dataframe.', required=True)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    df_folio = make_df(args.folio_df_path)
    df_nli = make_df(args.nli_df_path)
    
    df = pd.merge(df_folio, df_nli, on=['Sketcher'], suffixes=('_folio', '_nli'), how='outer')

    df.to_csv('evaluation/performance/performance_df_samples.csv', index=False)
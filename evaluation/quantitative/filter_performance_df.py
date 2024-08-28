
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
    group_cols = ['sketcher', 'refiner', 'gcd', 'finetune', 'baseline', 'use_backup']

    # Define the metrics to average
    metric_cols = ['f1', 'executable_f1', 'backup_f1_executable', 'backup_f1_non_executable' ,'executable_rate']

    # Group by the experiment columns and calculate the mean of the metrics
    df = df_original.groupby(group_cols).agg({col: ['mean', 'std'] for col in metric_cols}).reset_index()
    # make a new column new_f1 with the mean +- the std

    for col in ['f1', 'executable_f1', 'backup_f1_executable', 'backup_f1_non_executable']:
        df[col] = '$' + df[col, 'mean'].apply(lambda x: f'{np.round(x, 2)}') + '^{' + '\\pm ' + df[col, 'std'].apply(lambda y: f'{np.round(y, 2)}') + '}$'
        df.drop(columns=(col, 'std'), inplace=True)

    col = 'executable_rate'
    df[col] = '$' + df[col, 'mean'].apply(lambda x: f'{np.round(x, 4)*100}' if not np.isnan(x) else 'N/A') + '^{' + '\\pm' + df[col, 'std'].apply(lambda y: f'{int(y*100)}' if not np.isnan(y) else '') + '}' + '\\%$'
    df.drop(columns=(col, 'std'), inplace=True)
        
        
    df.columns = df.columns.droplevel(1)

    df_backup = df[df['use_backup'] == True]
    df_nobackup = df[df['use_backup'] == False]

    # merge the backup and no backup dataframes on the group columns
    df = pd.merge(df_backup, df_nobackup, on=group_cols, suffixes=('_backup', '_nobackup'), how='outer').drop(columns='use_backup')

    group_cols = ['sketcher', 'refiner', 'gcd', 'finetune', 'baseline']
    # group by group_cols to merge f1_backup and f1_nobackup
    df = df.groupby(group_cols).agg({'f1_backup': 'first', 
                                    'f1_nobackup': 'first',
                                    'executable_f1_backup': 'first',
                                    'executable_f1_nobackup': 'first',
                                    'executable_rate_backup': 'first',
                                    'backup_f1_executable_backup': 'first',
                                    'backup_f1_executable_nobackup': 'first',
                                    'backup_f1_non_executable_backup': 'first',
                                    'backup_f1_non_executable_nobackup': 'first'
                                    }).reset_index()


    df = df.drop(columns=['backup_f1_executable_backup', 'backup_f1_non_executable_backup', 'executable_f1_backup'])
    
    df.rename(columns={
        'backup_f1_executable_nobackup': 'backup_f1_executable', 
        'backup_f1_non_executable_nobackup': 'backup_f1_non_executable',
        'executable_f1_nobackup': 'executable_f1',
        'executable_rate_backup': 'executable_rate'
        }, inplace=True)
    
    df = df.rename(columns={'gcd': 'GCD'}).rename(columns={'finetune': 'Finetune'})
        
    # df['sketcher'] = df['sketcher'].apply(lambda x: change_cell(x))
    df['refiner'] = df['refiner'].apply(lambda x: rename_model(x))
    df['refiner'] = df['refiner'] + df['Finetune'].map({True: ' $\\blacklozenge$', False: ''})
    df['refiner'] = df['refiner'] + df['GCD'].map({True: ' $\\bigstar$', False: ''})
    
    df['sketcher'] = df['sketcher'].apply(lambda x: rename_model(x))
    
    
    df['strategy'] = df['baseline'].apply(lambda x: rename_strategy(x))
    
    df = df.drop(columns=['Finetune', 'GCD', 'baseline'])
    
    # df = df[['sketcher', 'strategy', 'refiner', 'f1_nobackup','executable_f1', 'executable_rate', 'f1_backup', 'backup_f1_executable', 'backup_f1_non_executable']].reset_index(drop=True)  
    
    df1 = df[['sketcher', 'strategy', 'refiner', 'f1_nobackup', 'f1_backup', 'executable_f1', 'executable_rate']].reset_index(drop=True)  
    
    df2 = df[['sketcher', 'strategy', 'refiner', 'executable_f1', 'backup_f1_executable', 'backup_f1_non_executable']].reset_index(drop=True)  
    
    # df2 = df[['sketcher', 'strategy', 'refiner', 'backup_f1_executable', 'backup_f1_non_executable']].reset_index(drop=True)
    
    
    df1 = df1.rename(columns={'f1_nobackup': 'Weighted F1\n(all samples)', 
                            'f1_backup': 'Weighted F1\n(all samples, with backup)',
                            'executable_f1': 'Weighted F1\n(executable samples)',
                            'executable_rate': 'Executable\nSamples (\\%)', 
                            'sketcher': 'Sketcher',
                            'refiner': 'Refiner',
                            'strategy': 'Strategy'})  
    
    df2 = df2.rename(columns={
                            'executable_f1': 'Weighted F1\n(executable samples)',
                            'backup_f1_executable': 'Weighted F1 of backup\n(executable samples)',
                            'backup_f1_non_executable': 'Weighted F1 of backup\n(non-executable samples)',
                            'sketcher': 'Sketcher',
                            'refiner': 'Refiner',
                            'strategy': 'Strategy'}) 

    
    return df1.reset_index(drop=True), df2.reset_index(drop=True)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Make a dataframe from the results csv file.')
    parser.add_argument('--folio_df_path', type=str, help='Path to the results csv file for FOLIO.', required=True)
    parser.add_argument('--nli_df_path', type=str, help='Path to the results csv file for LogicNLI.', required=True)
    # parser.add_argument('--save_path', type=str, help='Path to save the dataframe.', required=True)
    # parser.add_argument('--save_path_backup', type=str, help='Path to save the dataframe.', required=True)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    df_folio, df_folio_backup = make_df(args.folio_df_path)
    df_nli, df_nli_backup = make_df(args.nli_df_path)
    
    # df = pd.merge(df_folio, df_nli, on=['Sketcher', 'Strategy', 'Refiner'], suffixes=('_folio', '_nli'), how='outer')
    
    # make a multiindex for the columns: add a level for each dataset
    df_folio.columns = pd.MultiIndex.from_tuples([('FOLIO', col) for col in df_folio.columns])
    df_nli.columns = pd.MultiIndex.from_tuples([('LogicNLI', col) for col in df_nli.columns])
    
    # df_folio.to_csv('evaluation/performance/folio.csv', index=False)
    # df_nli.to_csv('evaluation/performance/nli.csv', index=False)

    df_folio.to_csv('fol.csv', index=False)
    df_nli.to_csv('nli.csv', index=False)
    
    # df_backup = pd.merge(df_folio_backup, df_nli_backup, on=['Sketcher', 'Strategy', 'Refiner'], suffixes=('_folio', '_nli'), how='outer')
    
    # df_backup.columns = pd.MultiIndex.from_tuples([('FOLIO', col.replace('_folio', '')) if 'folio' in col else ('LogicNLI', col.replace('_nli', '')) for col in df_backup.columns])
    
    # df_backup.to_csv('evaluation/performance/backup.csv', index=False)

    
    # # make a multiindex for the columns: add a level for each dataset
    # df_folio_backup.columns = pd.MultiIndex.from_tuples([('FOLIO', col) for col in df_folio_backup.columns])
    # df_nli_backup.columns = pd.MultiIndex.from_tuples([('FOLIO', col) for col in df_nli_backup.columns])
    
    # df_folio_backup.to_csv('evaluation/performance/folio_backup.csv', index=False)
    # df_nli_backup.to_csv('evaluation/performance/nli_backup.csv', index=False)
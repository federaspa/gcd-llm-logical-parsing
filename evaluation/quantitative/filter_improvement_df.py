
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
        return 'Chain-of-Thought'

    elif 'LogicLM' in x:
        return 'LogicLM'
    
    else:
        return 'GCLLM'

def make_df(df_path: str) -> pd.DataFrame:
    df_original = pd.read_csv(df_path)
    group_cols = ['sketcher', 'refiner']

    # Define the metrics to average
    metric_cols = ['fine\\_f1','fine\\_num','grammar\\_f1','grammar\\_num','grammar\\_fine\\_f1','grammar\\_fine\\_num','backup\\_f1','backup\\_num', 'random\\_f1', 'random\\_num', 'manual\\_f1', 'manual\\_num']

    # Group by the experiment columns and calculate the mean of the metrics
    df = df_original.groupby(group_cols).agg({col: ['mean', 'std'] for col in metric_cols}).reset_index()
    
    # make a new column new_f1 with the mean +- the std
    
    f1s = ['fine\\_f1', 'grammar\\_f1', 'grammar\\_fine\\_f1', 'backup\\_f1', 'random\\_f1', 'manual\\_f1']
    nums = ['fine\\_num', 'grammar\\_num', 'grammar\\_fine\\_num', 'backup\\_num', 'random\\_num', 'manual\\_num']

    for (col_f1, col_num) in zip(f1s, nums):
        df[col_f1] = '$' + df[col_f1, 'mean'].apply(lambda x: f'{np.round(x, 2)}') + '^{' + '\\pm ' + df[col_f1, 'std'].apply(lambda y: f'{np.round(y, 2)}') + '}' + '(' + df[col_num, 'mean'].apply(lambda z: f'{np.round(z, 2)}') + ')$'
        
        df.drop(columns=[(col_f1, 'std'), (col_num, 'mean'), (col_num, 'std')], inplace=True)
        
    df.columns = df.columns.droplevel(1)
    
    random_1 = df['random\\_f1'][0]
    random_2 = df['random\\_f1'][3]

    df = df[['sketcher', 'refiner', 'grammar\\_f1', 'manual\\_f1']]
    # duplicate every row by splitting the grammar_f1 and grammar_fine_f1
    df1 = df.copy()
    df2 = df.copy()
    df1['f1'] = df1['grammar\\_f1']
    df1['type'] = 'grammar'
    df2['f1'] = df2['manual\\_f1']
    df2['type'] = 'manual'

    df = pd.concat([df1, df2], ignore_index=True)
    df['refiner'] = df['refiner'] + df['type'].map({'grammar': ' $\\blacklozenge$ $\\bigstar$', 'manual': ' $\\bigstar$'})

    df.drop(columns=['grammar\\_f1', 'manual\\_f1', 'type'], inplace=True)
    df = df.sort_values(by=['sketcher', 'refiner']).reset_index(drop=True)
    df = pd.concat([df, pd.DataFrame([['gpt-3.5-turbo', 'random', random_1]], columns=df.columns), pd.DataFrame([['gpt-4o', 'random', random_2]], columns=df.columns)])
    
    return df.reset_index(drop=True)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Make a dataframe from the results csv file.')
    parser.add_argument('--folio_df_path', type=str, help='Path to the results csv file for FOLIO.', required=True)
    parser.add_argument('--nli_df_path', type=str, help='Path to the results csv file for LogicNLI.', required=True)
    parser.add_argument('--save_path', type=str, help='Path to save the dataframe.', required=True)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    df_folio = make_df(args.folio_df_path)
    df_nli = make_df(args.nli_df_path)
    
    df = pd.merge(df_folio, df_nli, on=['sketcher', 'refiner'], suffixes=('_folio', '_nli'), how='outer')
    
    # df = pd.concat([pd.DataFrame([['FOLIO']*len(df_folio.columns)], columns=df_folio.columns), df_folio, pd.DataFrame([['LogicNLI']*len(df_folio.columns)], columns=df_nli.columns), df_nli], axis=0)
    
    # make a multiindex for the columns: add a level for each dataset
    df.columns = pd.MultiIndex.from_tuples([('FOLIO', col.replace('_folio', '')) if 'folio' in col else ('LogicNLI', col.replace('_nli', '')) for col in df.columns])
    
    
    df.to_csv(args.save_path, index=False)
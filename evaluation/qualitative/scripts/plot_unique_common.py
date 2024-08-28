import json
import pandas as pd

with open('fixed_errors_folio.json', 'r') as f:
    fixed_folio = json.load(f)
    
with open('fixed_errors_nli.json', 'r') as f:
    fixed_nli = json.load(f)
    

folio_errors_2_7 = []
folio_errors_2_13 = []
folio_errors_3_8 = []

for key, errors in fixed_folio.items():
    if '_3.5' not in key or 'gpt' in key:
        continue
    
    if '7b' in key:
        folio_errors_2_7.extend([e['id'] for e in errors])
    elif '13b' in key:
        folio_errors_2_13.extend([e['id'] for e in errors])
    elif '8b' in key:
        folio_errors_3_8.extend([e['id'] for e in errors])
    

nli_errors_2_7 = []
nli_errors_2_13 = []
nli_errors_3_8 = []

for key, errors in fixed_nli.items():
    if '_3.5' not in key or 'gpt' in key:
        continue
    
    if '7b' in key:
        nli_errors_2_7.extend([e['id'] for e in errors])
    elif '13b' in key:
        nli_errors_2_13.extend([e['id'] for e in errors])
    elif '8b' in key:
        nli_errors_3_8.extend([e['id'] for e in errors])
    


# find unique and common errors
folio_errors_2_7 = set(folio_errors_2_7)
folio_errors_2_13 = set(folio_errors_2_13)
folio_errors_3_8 = set(folio_errors_3_8)

folio_common_errors = folio_errors_2_7.intersection(folio_errors_2_13).intersection(folio_errors_3_8)
folio_unique_errors_2_7 = folio_errors_2_7 - folio_errors_2_13 - folio_errors_3_8
folio_unique_errors_2_13 = folio_errors_2_13 - folio_errors_2_7 - folio_errors_3_8
folio_unique_errors_3_8 = folio_errors_3_8 - folio_errors_2_7 - folio_errors_2_13
    

# find unique and common errors
nli_errors_2_7 = set(nli_errors_2_7)
nli_errors_2_13 = set(nli_errors_2_13)
nli_errors_3_8 = set(nli_errors_3_8)

nli_common_errors = nli_errors_2_7.intersection(nli_errors_2_13).intersection(nli_errors_3_8)
nli_unique_errors_2_7 = nli_errors_2_7 - nli_errors_2_13 - nli_errors_3_8
nli_unique_errors_2_13 = nli_errors_2_13 - nli_errors_2_7 - nli_errors_3_8
nli_unique_errors_3_8 = nli_errors_3_8 - nli_errors_2_7 - nli_errors_2_13
    
# plot number of unique and common errors in a histogram with seaborn
import seaborn as sns
import matplotlib.pyplot as plt

data_folio = {
    'Llama 2 7b': len(folio_unique_errors_2_7),
    'Llama 2 13b': len(folio_unique_errors_2_13),
    'Llama 3 8b': len(folio_unique_errors_3_8),
    'Common': len(folio_common_errors)
}

df_folio = pd.DataFrame(data_folio, index=[0])
df_folio = df_folio.melt(var_name='Type', value_name='Count')

data_nli = {
    'Llama 2 7b': len(nli_unique_errors_2_7),
    'Llama 2 13b': len(nli_unique_errors_2_13),
    'Llama 3 8b': len(nli_unique_errors_3_8),
    'Common': len(nli_common_errors)
}

df_nli = pd.DataFrame(data_nli, index=[0])
df_nli = df_nli.melt(var_name='Type', value_name='Count')



plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
palette = sns.color_palette("Set2", n_colors=4)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax, df, name in zip(axes, [df_folio, df_nli], ['FOLIO', 'LogicNLI']):
    sns.barplot(x='Type', y='Count', data=df, palette=palette, hue='Type', ax=ax)
    ax.set_title('Number of unique and common errors\n{name}'.format(name=name))
    
plt.savefig('unique_common_errors.png')

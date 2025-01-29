import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Placeholder path to the CSV file
file_path = 'evaluator/evaluation_results.csv'

# Load the dataset
df = pd.read_csv(file_path)

# df = df[(df['Accuracy'] != -100) | (df['Coverage'] != -100)]
df["Model"] = df['Model'] + '-' + df["Size"].apply(lambda x: f"{int(x)}" if x == int(x) else f"{x}") + "b-it"

# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")

# Create a figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot Accuracy for each model, shots, and category
sns.barplot(x='Model', y='Accuracy', hue='Category', data=df, ax=axes[0])
axes[0].set_title('Accuracy by Model and Category')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# Plot Coverage for each model, shots, and category
sns.barplot(x='Model', y='Coverage', hue='Category', data=df, ax=axes[1], legend=False)
axes[1].set_title('Coverage by Model and Category')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.savefig('./here')
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_performance_changes(results):
    # Keep original order of models
    model_names = list(results.keys())

    # Create figure and axis
    plt.figure(figsize=(15, 10))

    # Set up positions for grouped bars
    num_models = len(model_names)
    bar_width = 0.15
    index = np.arange(num_models)

    # Plot bars for each format
    formats = ['UNCONSTRAINED', 'JSON', 'CONSTRAINED']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    for i, fmt in enumerate(formats):
        accuracies = [results[model][fmt]['accuracy'] * 100 for model in model_names]
        plt.bar(index + i*bar_width, accuracies, bar_width, 
                label=fmt.capitalize(), color=colors[i], alpha=0.8)

    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison Across Different Formats')
    plt.xticks(index + bar_width, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot coverage
    plt.subplot(2, 1, 2)
    for i, fmt in enumerate(formats):
        coverage = [results[model][fmt]['coverage'] * 100 for model in model_names]
        plt.bar(index + i*bar_width, coverage, bar_width, 
                label=fmt.capitalize(), color=colors[i], alpha=0.8)

    plt.ylabel('Coverage (%)')
    plt.title('Coverage Comparison Across Different Formats')
    plt.xticks(index + bar_width, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig('evaluation/plots/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Load and plot the results
with open('evaluation/evaluation_results/FOLIO_dev_results.json', 'r') as f:
    results = json.load(f)

plot_performance_changes(results)
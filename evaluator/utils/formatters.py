import pandas as pd
import numpy as np

class ResultFormatter:
    @staticmethod
    def create_dataframe(results):
        # Define columns
        columns = ['Model', 'Shots', 'Category', 'Accuracy', 'Coverage']
        
        # Process data
        data = []
        for model_name, model_results in sorted(results.items()):
            for shot in sorted(model_results.keys()):
                metrics = model_results[shot]
                for category in ['UNCONSTRAINED', 'FOL']:
                    data.append([
                        model_name,
                        f"{shot}-shots",
                        category,
                        metrics[category]['accuracy'][0] * 100,
                        metrics[category]['coverage'][0] * 100
                    ])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Format all numbers to two decimal places with percentage sign
        df['Accuracy'] = df['Accuracy'].map('{:.2f}'.format)
        df['Coverage'] = df['Coverage'].map('{:.2f}'.format)
        
        return df

    @staticmethod
    def format_latex(results, table_name=""):
        # Determine the number of shots dynamically
        shot_numbers = sorted(next(iter(results.values())).keys())
        
        # Create LaTeX table header
        latex_lines = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\begin{tabular}{|l|" + "V".join(["cc|cc" for _ in shot_numbers]) + "V}",
            "\\hline",
            "& " + " & ".join([f"\\multicolumn{{4}}{{cV}}{{\\textbf{{{shot}-shots}}}}" for shot in shot_numbers]) + " \\\\ \\hline",
            "\\textbf{Model} & " + " & ".join(["\\multicolumn{2}{c|}{\\textbf{Accuracy}} & \\multicolumn{2}{cV}{\\textbf{Coverage}}" for _ in shot_numbers]) + " \\\\ \\hline",
            "& " + " & ".join(["\\textit{Unc.} & \\textit{FOL} & \\textit{Unc.} & \\textit{FOL}" for _ in shot_numbers]) + " \\\\",
            "\\hline"
        ]
        
        previous_root = None
        # Process each model's results
        for model_name, model_results in sorted(results.items()):
            current_root = model_name.split('-')[0]
            formatted_metrics = [model_name]
            for shot in shot_numbers:
                metrics = model_results[shot]
                acc_values = [metrics['UNCONSTRAINED']['accuracy'][0], metrics['FOL']['accuracy'][0]]
                cov_values = [metrics['UNCONSTRAINED']['coverage'][0], metrics['FOL']['coverage'][0]]
                
                max_acc = max(acc_values) if max(acc_values) != 0 else np.inf
                max_cov = max(cov_values) if max(cov_values) != 0 else np.inf
                
                formatted_metrics.extend([
                    *[f"\\textbf{{{v:.2f}}}" if v == max_acc else f"{v:.2f}" for v in acc_values],
                    *[f"\\textbf{{{v:.2f}}}" if v == max_cov else f"{v:.2f}" for v in cov_values]
                ])
            
            if previous_root is not None and current_root != previous_root:
                latex_lines[-1] += "\\hline"
            
            latex_lines.append(" & ".join(formatted_metrics) + " \\\\")
            
            previous_root = current_root
        
        latex_lines[-1] += "\\hline"

        latex_lines.extend([
            "\\end{tabular}",
            f"\\caption{{{table_name}}}" if table_name else "",
            "\\label{{tab:combined}}",
            "\\end{table*}"
        ])
        
        return "\n".join(latex_lines)
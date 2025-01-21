import pandas as pd

class ResultFormatter:
    @staticmethod
    def create_dataframe(results):
        # Define MultiIndex for columns
        categories = ['Unconstrained', 'JSON', 'Constrained']
        column_tuples = [(cat, metric) for cat in categories for metric in ['Accuracy', 'Coverage']]
        columns = pd.MultiIndex.from_tuples(column_tuples)
        
        # Process data
        data = {}
        for model_name, model_results in sorted(results.items()):
            row_data = []
            for category in ['UNCONSTRAINED', 'JSON', 'CONSTRAINED']:
                metrics = model_results[category]
                row_data.extend([
                    metrics[0][0] if isinstance(metrics[0], tuple) else metrics[0],
                    metrics[3][0] if isinstance(metrics[3], tuple) else metrics[3]
                ])
            data[model_name] = row_data
        
        # Create and format DataFrame
        df = pd.DataFrame(data).T
        df.columns = columns
        df = df.round(4) * 100
        
        # Format all numbers to two decimal places with percentage sign
        pd.options.display.float_format = '{:.2f}%'.format
        
        return df

    @staticmethod
    def format_latex(results, table_name=""):
        latex_lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{l|ccc|ccc}",
            "\\hline",
            "Model & \\multicolumn{3}{c|}{Accuracy} & \\multicolumn{3}{c}{Coverage}\\\\ \\hline ",
            " & Unc & Json & Con & Unc & Json & Con \\\\",
            "\\hline"
        ]
        
        # Process each model's results
        for model_name, model_results in sorted(results.items()):
            metrics = {
                'unc': model_results['UNCONSTRAINED'],
                'json': model_results['JSON'],
                'con': model_results['CONSTRAINED']
            }
            
            # Get max values for highlighting
            acc_values = [m[0][0] for m in metrics.values()]
            cov_values = [m[3][0] for m in metrics.values()]
            max_acc = max(acc_values)
            max_cov = max(cov_values)
            
            # Format line
            formatted_metrics = [
                model_name,
                *[f"\\textbf{{{v:.2f}}}" if v == max_acc else f"{v:.2f}" 
                  for v in acc_values],
                *[f"\\textbf{{{v:.2f}}}" if v == max_cov else f"{v:.2f}"
                  for v in cov_values]
            ]
            
            latex_lines.append(" & ".join(formatted_metrics) + " \\\\" + " \\hline")
        
        latex_lines.extend([
            "\\end{tabular}",
            f"\\caption{{{table_name}}}" if table_name else "",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)
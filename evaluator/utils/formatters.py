import pandas as pd
import numpy as np

class ResultFormatter:
    @staticmethod
    def create_dataframe(results):
        # Define columns
        columns = ['Model', 'Size', 'Shots', 'Category', 'Accuracy', 'Accuracy_improvement', 'Coverage', 'Coverage_improvement']
        
        # Process data
        data = []
        for model_name, model_results in sorted(results.items()):
            # Split model_name into Model and Size
            model_parts = model_name.split('-')
            model = model_parts[0]
            size = model_parts[1].replace('b', '')  # Remove 'b' to sort numerically
            
            for shot in sorted(model_results.keys()):
                metrics = model_results[shot]
                for category in ['UNCONSTRAINED', 'FOL']:
                    data.append([
                        model,
                        size,
                        f"{shot}-shots",
                        category,
                        round(metrics[category]['accuracy'] * 100, 2),
                        round((metrics['FOL']['accuracy'] - metrics['UNCONSTRAINED']['accuracy']) * 100, 2),
                        round(metrics[category]['coverage'] * 100, 2),
                        round((metrics['FOL']['coverage'] - metrics['UNCONSTRAINED']['coverage']) * 100, 2)
                    ])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Convert Size to numeric for sorting
        df['Size'] = pd.to_numeric(df['Size'])
        
        # Sort by Model and Size
        df = df.sort_values(by=['Model', 'Size'])
        
        return df

    @staticmethod
    def format_latex(results):
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
        
        # Process each model's results
        for model_name, model_results in sorted(results.items()):
            formatted_metrics = [model_name]
            for shot in shot_numbers:
                metrics = model_results[shot]
                improvement_accuracy = metrics['FOL']['accuracy']- metrics['UNCONSTRAINED']['accuracy']
                improvement_coverage = metrics['FOL']['coverage']- metrics['UNCONSTRAINED']['coverage']

                
                acc_values = [metrics['UNCONSTRAINED']['accuracy'], metrics['FOL']['accuracy']]
                cov_values = [metrics['UNCONSTRAINED']['coverage'], metrics['FOL']['coverage']]
                
                max_acc = max(acc_values) if max(acc_values) > 0 else np.inf
                max_cov = max(cov_values) if max(cov_values) > 0 else np.inf
                
                # Format base values with improvements
                formatted_values = []
                
                # Format unconstrained accuracy (no improvement shown)
                formatted_values.append(
                    f"\\textbf{{{acc_values[0]:.2f}}}" if acc_values[0] == max_acc else f"{acc_values[0]:.2f}"
                )
                
                # Format FOL accuracy with improvement
                fol_acc = f"\\textbf{{{acc_values[1]:.2f}}}" if acc_values[1] == max_acc else f"{acc_values[1]:.2f}"
                fol_acc += f" ({improvement_accuracy:.2f})"
                formatted_values.append(fol_acc)
                
                # Format unconstrained coverage (no improvement shown)
                formatted_values.append(
                    f"\\textbf{{{cov_values[0]:.2f}}}" if cov_values[0] == max_cov else f"{cov_values[0]:.2f}"
                )
                
                # Format FOL coverage with improvement
                fol_cov = f"\\textbf{{{cov_values[1]:.2f}}}" if cov_values[1] == max_cov else f"{cov_values[1]:.2f}"
                fol_cov += f" ({improvement_coverage:.2f})"
                formatted_values.append(fol_cov)
                
                formatted_metrics.extend(formatted_values)
            
            latex_lines.append(" & ".join(formatted_metrics) + " \\\\")
        
        latex_lines.extend([
            "\\end{tabular}",
            "\\caption{}",
            "\\label{{tab:combined}}",
            "\\end{table*}"
        ])
        
        return "\n".join(latex_lines)
    
    
    @staticmethod 
    def format_latex_split(results):
        # Determine the number of shots dynamically
        shot_numbers = sorted(next(iter(results.values())).keys())
        
        # Create LaTeX table header
        accuracy_lines = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\begin{tabular}{|l|" + "V".join(["cc" for _ in shot_numbers]) + "V}",
            "\\hline",
            "\\textbf{Model} & " + " & ".join([f"\\multicolumn{{2}}{{cV}}{{\\textbf{{{shot}-shots}}}}" for shot in shot_numbers]) + " \\\\ \\hline",
            # "\\textbf{Model} & " + " & ".join(["\\multicolumn{2}{c|}{\\textbf{Accuracy}}" for _ in shot_numbers]) + " \\\\ \\hline",
            "& " + " & ".join(["\\textit{Unc.} & \\textit{FOL}" for _ in shot_numbers]) + " \\\\",
            "\\hline"
        ]
        coverage_lines = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\begin{tabular}{|l|" + "V".join(["cc" for _ in shot_numbers]) + "V}",
            "\\hline",
            "\\textbf{Model} & " + " & ".join([f"\\multicolumn{{2}}{{cV}}{{\\textbf{{{shot}-shots}}}}" for shot in shot_numbers]) + " \\\\ \\hline",
            # "\\textbf{Model} & " + " & ".join(["\\multicolumn{2}{c|}{\\textbf{Accuracy}}" for _ in shot_numbers]) + " \\\\ \\hline",
            "& " + " & ".join(["\\textit{Unc.} & \\textit{FOL}" for _ in shot_numbers]) + " \\\\",
            "\\hline"
        ]
        
        # Process each model's results
        for model_name, model_results in sorted(results.items()):
            formatted_accuracy = [model_name]
            formatted_coverage = [model_name]
            for shot in shot_numbers:
                metrics = model_results[shot]
                improvement_accuracy = metrics['FOL']['accuracy']- metrics['UNCONSTRAINED']['accuracy']
                improvement_coverage = metrics['FOL']['coverage']- metrics['UNCONSTRAINED']['coverage']

                
                acc_values = [metrics['UNCONSTRAINED']['accuracy'], metrics['FOL']['accuracy']]
                cov_values = [metrics['UNCONSTRAINED']['coverage'], metrics['FOL']['coverage']]
                
                max_acc = max(acc_values) if max(acc_values) > 0 else np.inf
                max_cov = max(cov_values) if max(cov_values) > 0 else np.inf
                
                # Format base values with improvements
                formatted_acc = []
                formatted_cov = []
                
                # Format unconstrained accuracy (no improvement shown)
                formatted_acc.append(
                    f"\\textbf{{{acc_values[0]:.2f}}}" if acc_values[0] == max_acc else f"{acc_values[0]:.2f}"
                )
                
                # Format FOL accuracy with improvement
                fol_acc = f"\\textbf{{{acc_values[1]:.2f}}}" if acc_values[1] == max_acc else f"{acc_values[1]:.2f}"
                fol_acc += f" ({improvement_accuracy:.2f})"
                formatted_acc.append(fol_acc)
                
                # Format unconstrained coverage (no improvement shown)
                formatted_cov.append(
                    f"\\textbf{{{cov_values[0]:.2f}}}" if cov_values[0] == max_cov else f"{cov_values[0]:.2f}"
                )
                
                # Format FOL coverage with improvement
                fol_cov = f"\\textbf{{{cov_values[1]:.2f}}}" if cov_values[1] == max_cov else f"{cov_values[1]:.2f}"
                fol_cov += f" ({improvement_coverage:.2f})"
                formatted_cov.append(fol_cov)
                
                formatted_accuracy.extend(formatted_acc)
                formatted_coverage.extend(formatted_cov)
            
            accuracy_lines.append(" & ".join(formatted_accuracy) + " \\\\")
            coverage_lines.append(" & ".join(formatted_coverage) + " \\\\")
            
        accuracy_lines.extend([
            "\\end{tabular}",
            "\\caption{Accuracy}",
            "\\label{tab:acc}",
            "\\end{table*}"
        ])
        
        coverage_lines.extend([
            "\\end{tabular}",
            "\\caption{Coverage}",
            "\\label{tab:cov}",
            "\\end{table*}"
        ])
        
        return "\n".join(accuracy_lines) + "\n\n" + "\n".join(coverage_lines)
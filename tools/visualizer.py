from graphviz import Digraph
import re
import json
from typing import Dict, List, Tuple, Optional, Union, Set
from pathlib import Path

class LogicTreeVisualizer:
    """A class to visualize logical reasoning trees from structured data."""
    
    def __init__(self):
        """Initialize the visualizer with default styling settings."""
        self.colors = {
            'contradiction': '#ffcccc',      # Light red
            'entailment': '#ccffcc',         # Light green
            'neutral': '#e6e6e6',            # Light gray
            'self_contradiction': '#ffb366'   # Light orange
        }
        
        self.node_defaults = {
            'shape': 'box',
            'style': 'rounded'
        }
        
        self.graph_defaults = {
            'rankdir': 'BT',  # Bottom to Top direction
            'splines': 'ortho'  # Orthogonal lines for cleaner appearance
        }
    
    def _clean_node_text(self, text: str) -> str:
        """Clean node text for better visualization."""
        return str(text).replace('"', '').replace("'", "")
    
    def _parse_reasoning_path(self, path: str) -> List[Tuple[str, str]]:
        """Parse a reasoning path into individual connections."""
        if not path or path == 'none':
            return []
            
        # Split by operators while keeping them
        parts = re.split(r'(\+|\||\-->)', path)
        parts = [p.strip() for p in parts if p.strip()]
        
        # Group facts and operators
        connections = []
        for i in range(len(parts)):
            if parts[i] == '-->':
                if i > 0 and i < len(parts) - 1:
                    source = parts[i-1].strip('[]')
                    target = parts[i+1].strip('[]')
                    connections.append((source, target))
        
        return connections
    
    def _get_node_dependencies(self, stmt_id: str, statements: Dict, facts: Dict) -> Set[str]:
        """Get all nodes (facts and statements) that a statement depends on."""
        dependencies = set()
        
        def process_reasoning(reasoning: str):
            if not reasoning or reasoning == 'none':
                return
            
            # Extract fact and statement references
            matches = re.findall(r'fact \d+|stmt \d+', reasoning)
            for match in matches:
                dependencies.add(match)
                
                # If it's a statement, recursively process its dependencies
                if match.startswith('stmt '):
                    stmt_num = match.split()[1]
                    if stmt_num in statements:
                        process_reasoning(statements[stmt_num][3])
        
        # Process the statement's reasoning
        if stmt_id in statements:
            process_reasoning(statements[stmt_id][3])
        
        return dependencies

    def create_statement_tree(self, data: Dict, stmt_id: str) -> Digraph:
        """Create a visualization tree for a single statement.
        
        Args:
            data: Dictionary containing the logic problem data
            stmt_id: ID of the statement to visualize
            
        Returns:
            Graphviz Digraph object
        """
        # Create a new directed graph
        dot = Digraph(comment=f'Logic Tree - Statement {stmt_id}')
        
        # Apply graph defaults
        dot.attr('graph', **self.graph_defaults)
        
        # Apply node defaults
        dot.attr('node', **self.node_defaults)
        
        statements = data.get('statements', {})
        facts = data.get('facts', {})
        labels = data.get('labels', {})
        
        if stmt_id not in statements:
            return dot
        
        # Get all dependencies for this statement
        dependencies = self._get_node_dependencies(stmt_id, statements, facts)
        
        # Add the main statement node
        stmt_info = statements[stmt_id]
        main_node_id = f"stmt {stmt_id}"
        label = f"Statement {stmt_id}\n{stmt_info[0]} is {stmt_info[1]} ({stmt_info[2]})"
        fill_color = self.colors.get(labels.get(stmt_id, 'neutral'), '#ffffff')
        dot.node(main_node_id, label, fillcolor=fill_color, style='filled,rounded')
        
        # Add dependent fact nodes
        for dep in dependencies:
            if dep.startswith('fact '):
                fact_id = dep.split()[1]
                if fact_id in facts:
                    fact_info = facts[fact_id]
                    node_id = f"fact {fact_id}"
                    label = f"Fact {fact_id}\n{fact_info[0]} is {fact_info[1]} ({fact_info[2]})"
                    dot.node(node_id, label)
        
        # Add dependent statement nodes
        for dep in dependencies:
            if dep.startswith('stmt '):
                dep_id = dep.split()[1]
                if dep_id in statements:
                    stmt_info = statements[dep_id]
                    node_id = f"stmt {dep_id}"
                    label = f"Statement {dep_id}\n{stmt_info[0]} is {stmt_info[1]} ({stmt_info[2]})"
                    fill_color = self.colors.get(labels.get(dep_id, 'neutral'), '#ffffff')
                    dot.node(node_id, label, fillcolor=fill_color, style='filled,rounded')
        
        # Add edges based on reasoning
        reasoning = stmt_info[3]
        connections = self._parse_reasoning_path(reasoning)
        
        for source, target in connections:
            # Clean up source and target IDs
            source_id = f"fact {source}" if source.startswith('fact') else f"stmt {source}"
            target_id = f"fact {target}" if target.startswith('fact') else f"stmt {target}"
            
            # Add edge
            dot.edge(source_id, target_id)
        
        return dot
    
    def visualize_statements(self, 
                           data: Dict, 
                           output_dir: str = 'logic_trees',
                           formats: List[str] = ['pdf', 'png']) -> None:
        """Create and save visualizations for each statement.
        
        Args:
            data: Dictionary containing the logic problem data
            output_dir: Directory to save output files
            formats: List of output formats to generate
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create a tree for each statement
        statements = data.get('statements', {})
        for stmt_id in statements:
            tree = self.create_statement_tree(data, stmt_id)
            base_path = f"{output_dir}/statement_{stmt_id}"
            
            for fmt in formats:
                tree.render(base_path, format=fmt, cleanup=True)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'LogicTreeVisualizer':
        """Create a LogicTreeVisualizer instance and load data from a JSON file."""
        visualizer = cls()
        with open(json_path, 'r') as f:
            visualizer.data = json.load(f)
        return visualizer
    
    def customize_style(self, 
                       colors: Optional[Dict[str, str]] = None,
                       node_defaults: Optional[Dict[str, str]] = None,
                       graph_defaults: Optional[Dict[str, str]] = None) -> None:
        """Customize the visualization styling."""
        if colors:
            self.colors.update(colors)
        if node_defaults:
            self.node_defaults.update(node_defaults)
        if graph_defaults:
            self.graph_defaults.update(graph_defaults)
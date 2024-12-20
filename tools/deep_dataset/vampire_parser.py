from collections import defaultdict
import re
import networkx as nx

class VampireParser:
    def __init__(self):
        # Store clauses and their derivation information
        self.clauses = {}
        # Track dependencies between clauses
        self.dependencies = defaultdict(list)
        # Graph to calculate derivation steps
        self.graph = nx.DiGraph()
        
    def extract_relevant_lines(self, vampire_output):
        """Extract relevant [SA] new lines from the last successful strategy."""
        current_lines = []
        last_successful_lines = []
        
        for line in vampire_output.split('\n'):
            if line.startswith('% Running in auto'):
                current_lines = []
            elif '[SA] new:' in line:
                current_lines.append(line)
            elif 'Success in time' in line or 'Satisfiable' in line:
                if current_lines:
                    last_successful_lines = current_lines
        
        return '\n'.join(last_successful_lines)

    def parse_line(self, line):
        """Parse a single line of Vampire output."""
        # Match the basic structure of a Vampire output line
        match = re.match(r'\[SA\] new: (\d+)\. (.*?) \[(.*?)\]', line)
        if not match:
            return None
            
        clause_num, clause_content, derivation = match.groups()
        clause_num = int(clause_num)
        
        # Store the clause information
        self.clauses[clause_num] = {
            'content': clause_content,
            'derivation_type': derivation.split()[0],
            'source': []
        }
        
        # Handle different types of derivations
        if 'cnf transformation' in derivation:
            source_num = int(re.search(r'cnf transformation (\d+)', derivation).group(1))
            self.clauses[clause_num]['source'] = [source_num]
            self.graph.add_edge(source_num, clause_num)
            
        elif 'resolution' in derivation:
            # Extract source clause numbers for resolution
            x = derivation.split('resolution')[1]
            source_nums = re.findall(r'\d+', x)
            source_nums = [int(num) for num in source_nums]
            self.clauses[clause_num]['source'] = source_nums
            
            # Add edges to the dependency graph
            for source in source_nums:
                self.graph.add_edge(source, clause_num)
                
        return clause_num
    
    def calculate_steps(self, clause_num):
        """Calculate minimum steps required to derive a clause."""
        # Find all source nodes (CNF transformations)
        source_nodes = [n for n in self.graph.nodes() 
                       if self.graph.in_degree(n) == 0]
        
        # If the clause doesn't exist in our graph, return None
        if clause_num not in self.graph:
            return None
            
        # Calculate the longest path from any source to this clause
        max_steps = 0
        for source in source_nodes:
            try:
                path_length = nx.shortest_path_length(self.graph, source, clause_num)
                max_steps = max(max_steps, path_length)
            except nx.NetworkXNoPath:
                continue
                
        return max_steps
    
    def parse_output(self, vampire_output):
        """Parse complete Vampire output and analyze all clauses."""
        results = {}
        
        # Process each line
        for line in vampire_output.split('\n'):
            if '[SA] new:' in line:
                clause_num = self.parse_line(line)
                if clause_num is not None:
                    steps = self.calculate_steps(clause_num)
                    results[clause_num] = {
                        'content': self.clauses[clause_num]['content'],
                        'steps': steps,
                        'derivation': self.clauses[clause_num]['derivation_type'],
                        'source': self.clauses[clause_num]['source']
                    }
        
        return results

# Example usage
def analyze_vampire_output(output_text):
    parser = VampireParser()
    results = parser.parse_output(output_text)
    
    print("Analysis of derivation steps:")
    for clause_num, info in sorted(results.items()):
        print(f"\nClause {clause_num}:")
        print(f"Content: {info['content']}")
        print(f"Steps required: {info['steps']}")
        print(f"Derived via: {info['derivation']}")
        print(f"Source clauses: {info['source']}")

# Example with your provided output
vampire_output = """
% Running in auto input_syntax mode. Trying TPTP
% WARNING: time unlimited strategy and instruction limiting not in place - attempting to translate instructions to time
% lrs+21_1:32_anc=all:to=lpo:sil=256000:plsq=on:plsqr=32,1:sp=occurrence:sos=on:plsql=on:sac=on:newcnf=on:i=222662:add=off:fsr=off:rawr=on_0 on tmprphz3u_6 for (99ds/222662Mi)
[SA] new: 10. likes(alan,bob) [cnf transformation 3]
[SA] new: 11. is(alan,round) [cnf transformation 2]
[SA] new: 12. is(alan,young) [cnf transformation 1]
[SA] new: 13. ~likes(X0,bob) | ~is(X0,kind) | is(X0,short) [cnf transformation 7]
[SA] new: 14. ~is(X0,young) | ~is(X0,round) | is(X0,kind) [cnf transformation 9]
% Refutation not found, incomplete strategy% ------------------------------
% Version: Vampire 4.9 (commit 5ad494e78 on 2024-06-14 14:05:27 +0100)
% Linked with Z3 4.12.3.0 79bbbf76d0c123481c8ca05cd3a98939270074d3 z3-4.8.4-7980-g79bbbf76d
% Termination reason: Refutation not found, incomplete strategy

% Memory used [KB]: 1039
% Time elapsed: 0.002 s
% ------------------------------
% ------------------------------
% Running in auto input_syntax mode. Trying TPTP
% WARNING: time unlimited strategy and instruction limiting not in place - attempting to translate instructions to time
% lrs+1011_4:1_sil=256000:rp=on:newcnf=on:i=257909:aac=none:gsp=on_0 on tmprphz3u_6 for (99ds/257909Mi)
[SA] new: 15. ~likes(alan,bob) [consistent polarity flipping 10]
[SA] new: 11. is(alan,round) [cnf transformation 2]
[SA] new: 12. is(alan,young) [cnf transformation 1]
[SA] new: 16. likes(X0,bob) | ~is(X0,kind) | is(X0,short) [consistent polarity flipping 13]
[SA] new: 14. ~is(X0,young) | ~is(X0,round) | is(X0,kind) [cnf transformation 9]
% Refutation not found, incomplete strategy% ------------------------------
% Version: Vampire 4.9 (commit 5ad494e78 on 2024-06-14 14:05:27 +0100)
% Linked with Z3 4.12.3.0 79bbbf76d0c123481c8ca05cd3a98939270074d3 z3-4.8.4-7980-g79bbbf76d
% Termination reason: Refutation not found, incomplete strategy

% Memory used [KB]: 1047
% Time elapsed: 0.001 s
% ------------------------------
% ------------------------------
% Running in auto input_syntax mode. Trying TPTP
% WARNING: time unlimited strategy and instruction limiting not in place - attempting to translate instructions to time
% dis+1002_1:1_tgt=full:sos=on:rp=on:sac=on:i=258102:ss=axioms:sd=3:cond=fast:add=off:abs=on:fde=none:sil=256000_0 on tmprphz3u_6 for (99ds/258102Mi)
% Refutation not found, incomplete strategy% ------------------------------
% Version: Vampire 4.9 (commit 5ad494e78 on 2024-06-14 14:05:27 +0100)
% Linked with Z3 4.12.3.0 79bbbf76d0c123481c8ca05cd3a98939270074d3 z3-4.8.4-7980-g79bbbf76d
% Termination reason: Refutation not found, incomplete strategy

% Memory used [KB]: 1038
% Time elapsed: 0.002 s
% ------------------------------
% ------------------------------
% Running in auto input_syntax mode. Trying TPTP
% WARNING: time unlimited strategy and instruction limiting not in place - attempting to translate instructions to time
% lrs+21_8:1_to=lpo:sil=2000:sp=frequency:spb=units:s2a=on:s2pl=no:i=103:sd=2:ss=included:fsr=off:fs=off_0 on tmprphz3u_6 for (1ds/103Mi)
[SA] new: 10. likes(alan,bob) [cnf transformation 3]
[SA] new: 11. is(alan,round) [cnf transformation 2]
[SA] new: 12. is(alan,young) [cnf transformation 1]
[SA] new: 13. is(X0,short) | ~is(X0,kind) | ~likes(X0,bob) [cnf transformation 7]
[SA] new: 14. is(X0,kind) | ~is(X0,round) | ~is(X0,young) [cnf transformation 9]
[SA] new: 15. ~is(alan,kind) | is(alan,short) [resolution 13,10]
[SA] new: 23. ~is(alan,kind) <- (~2) [avatar component clause 21]
[SA] new: 25. is(alan,kind) | ~is(alan,young) [resolution 14,11]
[SA] new: 29. ~is(alan,young) <- (~3) [avatar component clause 27]
[SA] new: 31. $false <- (~3) [resolution 29,12]
[SA] new: 28. is(alan,young) <- (3) [avatar component clause 27]
[SA] new: 22. is(alan,kind) <- (2) [avatar component clause 21]
[SA] new: 19. is(alan,short) <- (1) [avatar component clause 17]
% Solution written to "/tmp/vampire-proof-3264483"
% Running in auto input_syntax mode. Trying TPTP
% SZS status Satisfiable for tmprphz3u_6
% # SZS output start Saturation.
cnf(u19,axiom,
    is(alan,short)).

cnf(u22,axiom,
    is(alan,kind)).

cnf(u28,axiom,
    is(alan,young)).

cnf(u13,axiom,
    ~likes(X0,bob) | ~is(X0,kind) | is(X0,short)).

cnf(u12,axiom,
    is(alan,young)).

cnf(u14,axiom,
    ~is(X0,round) | is(X0,kind) | ~is(X0,young)).

cnf(u11,axiom,
    is(alan,round)).

cnf(u10,axiom,
    likes(alan,bob)).

% # SZS output end Saturation.
% ------------------------------
% Version: Vampire 4.9 (commit 5ad494e78 on 2024-06-14 14:05:27 +0100)
% Linked with Z3 4.12.3.0 79bbbf76d0c123481c8ca05cd3a98939270074d3 z3-4.8.4-7980-g79bbbf76d
% Termination reason: Satisfiable

% Memory used [KB]: 1056
% Time elapsed: 0.001 s
% ------------------------------
% ------------------------------
% Success in time 0.022 s
"""

if __name__ == "__main__":
    analyze_vampire_output(vampire_output)
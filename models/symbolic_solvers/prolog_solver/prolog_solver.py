def convert_to_prolog(input_text):
    lines = input_text.strip().split('\n')
    prolog_rules = []
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line in ['Predicates:', 'Facts:', 'Rules:', 'Query:']:
            current_section = line[:-1]
            continue
            
        if current_section == 'Facts':
            # Convert facts: Predicate(X, True) -> predicate(x).
            pred, args = line.split('(', 1)
            args = args[:-1].split(',')  # Remove trailing ) and split
            if args[1].strip() == 'True':
                fact = f"{pred.lower()}({args[0].lower()})."
                prolog_rules.append(fact)
                
        elif current_section == 'Rules':
            # Convert rules: A($x, True) >>> B($x, True) -> a(X) :- b(X).
            if '>>>' in line:
                antecedent, consequent = line.split('>>>')
                ant_pred, ant_args = antecedent.strip().split('(', 1)
                cons_pred, cons_args = consequent.strip().split('(', 1)
                
                # Extract variable name
                var_name = ant_args.split(',')[0].strip()
                rule = f"{ant_pred.lower()}({var_name.lower()}) :- {cons_pred.lower()}({var_name.lower()})."
                prolog_rules.append(rule)
                
        elif current_section == 'Query':
            # Convert query: Predicate(X, False) -> ?- \+predicate(x).
            pred, args = line.split('(', 1)
            args = args[:-1].split(',')
            if args[1].strip() == 'False':
                query = f"?- \\+{pred.lower()}({args[0].lower()})."
            else:
                query = f"?- {pred.lower()}({args[0].lower()})."
            prolog_rules.append('\n' + query)

    return '\n'.join(prolog_rules)

# Test with the given input
input_text = """
Predicates:
Jompus($x, bool) ::: Does x belong to Jompus?
Fruity($x, bool) ::: Is x fruity?
Wumpus($x, bool) ::: Does x belong to Wumpus?
Transparent($x, bool) ::: Is x transparent?
Tumpuses($x, bool) ::: Does x belong to Tumpuses?
Mean($x, bool) ::: Is x mean?
Vumpuses($x, bool) ::: Does x belong to Vumpuses?
Cold($x, bool) ::: Is x cold?
Yumpus($x, bool) ::: Does x belong to Yumpus?
Orange($x, bool) ::: Is x orange?
Numpus($x, bool) ::: Does x belong to Numpus?
Dull($x, bool) ::: Is x dull?
Dumpus($x, bool) ::: Does x belong to Dumpus?
Shy($x, bool) ::: Is x shy?
Impuses($x, bool) ::: Does x belong to Impuses?
Rompus($x, bool) ::: Does x belong to Rompus?
Liquid($x, bool) ::: Is x liquid?
Zumpus($x, bool) ::: Does x belong to Zumpus?
Facts:
Tumpuses(Alex, True)
Rules:
Jompus($x, True) >>> Fruity($x, True)
Jompus($x, True) >>> Wumpus($x, True)
Wumpus($x, True) >>> Transparent($x, False)
Wumpuses($x, True) >>> Tumpuses($x, True)
Tumpuses($x, True) >>> Mean($x, True)
Tumpuses($x, True) >>> Vumpuses($x, True)
Vumpuses($x, True) >>> Cold($x, True)
Vumpuses($x, True) >>> Yumpus($x, True)
Yumpus($x, True) >>> Orange($x, True)
Yumpus($x, True) >>> Numpus($x, True)
Numpus($x, True) >>> Dull($x, True)
Numpus($x, True) >>> Dumpus($x, True)
Dumpus($x, True) >>> Shy($x, False)
Impuses($x, True) >>> Shy($x, True)
Dumpus($x, True) >>> Rompus($x, True)
Rompus($x, True) >>> Liquid($x, True)
Rompus($x, True) >>> Zumpus($x, True)
Query:
Shy(Alex, False)
"""

print(convert_to_prolog(input_text))
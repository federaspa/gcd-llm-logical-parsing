from z3 import *

## Declarations
workers_sort, (George, Helena, Inga, Kelly, Leanda, Maricita, Olaf) = EnumSort('workers', ['George', 'Helena', 'Inga', 'Kelly', 'Leanda', 'Maricita', 'Olaf'])
tasks_sort, (framing, wallboarding, taping, sanding, priming) = EnumSort('tasks', ['framing', 'wallboarding', 'taping', 'sanding', 'priming'])
day_sort = IntSort()
workers = [George, Helena, Inga, Kelly, Leanda, Maricita, Olaf]
tasks = [framing, wallboarding, taping, sanding, priming]
day = [1, 2, 3]
assigned = Function('assigned', workers_sort, day_sort, tasks_sort, BoolSort())

## Constraints

pre_conditions = []
pre_conditions.append(And([Sum([assigned(w, d, t) for t in tasks for w in workers]) >= 1 for d in day]))
t = Const('t', tasks_sort)
pre_conditions.append(ForAll([t], Distinct([assigned(w, d, t) for d in day])))
w = Const('w', workers_sort)
pre_conditions.append(ForAll([w], Sum([assigned(w, d, t) for t in tasks for d in day]) >= 1))
w = Const('w', workers_sort)
pre_conditions.append(ForAll([w], Sum([assigned(w, d, t) for t in tasks for d in day]) <= 1))
pre_conditions.append(Distinct([assigned(w, d, t) for t in tasks]))
pre_conditions.append(assigned(George, _, taping))
pre_conditions.append(assigned(Helena, _, priming))
pre_conditions.append(assigned(Inga, _, framing))
pre_conditions.append(assigned(Kelly, _, framing))
pre_conditions.append(assigned(Leanda, _, wallboarding))
pre_conditions.append(assigned(Maricita, _, sanding))
pre_conditions.append(assigned(Olaf, _, wallboarding))
pre_conditions.append(assigned(w, _, t) == true implies assigned(w, _, t) == true)
pre_conditions.append(assigned(Maricita, 3, sanding))
pre_conditions.append(And([Sum([assigned(w, d, t) for t in tasks for w in workers]) >= 1 for d in day]))
t = Const('t', tasks_sort)
pre_conditions.append(ForAll([t], Distinct([assigned(w, d, t) for d in day])))
w = Const('w', workers_sort)
pre_conditions.append(ForAll([w], Sum([assigned(w, d, t) for t in tasks for d in day]) >= 1))
w = Const('w', workers_sort)
pre_conditions.append(ForAll([w], Sum([assigned(w, d, t) for t in tasks for d in day]) <= 1))
pre_conditions.append(Distinct([assigned(w, d, t) for t in tasks]))
pre_conditions.append(assigned(George, _, taping))
pre_conditions.append(assigned(Helena, _, priming))
pre_conditions.append(assigned(Inga, _, framing))
pre_conditions.append(assigned(Kelly, _, framing))
pre_conditions.append(assigned(Leanda, _, wallboarding))
pre_conditions.append(assigned(Maricita, _, sanding))
pre_conditions.append(assigned(Olaf, _, wallboarding))
pre_conditions.append(assigned(w, _, t) == true implies assigned(w, _, t) == true)
pre_conditions.append(assigned(Maricita, 3, sanding))

def is_valid(option_constraints):
    solver = Solver()
    solver.add(pre_conditions)
    solver.add(Not(option_constraints))
    return solver.check() == unsat

def is_unsat(option_constraints):
    solver = Solver()
    solver.add(pre_conditions)
    solver.add(option_constraints)
    return solver.check() == unsat

def is_sat(option_constraints):
    solver = Solver()
    solver.add(pre_conditions)
    solver.add(option_constraints)
    return solver.check() == sat

def is_accurate_list(option_constraints):
    return is_valid(Or(option_constraints)) and all([is_sat(c) for c in option_constraints])

def is_exception(x):
    return not x


## Options

if is_valid(assigned(Inga, 2, _)): print('(A)')
if is_valid(assigned(Kelly, 2, _)): print('(B)')
if is_valid(assigned(Olaf, 2, _)): print('(C)')
if is_valid(assigned(George, 2, _) and assigned(Helena, 2, _)): print('(D)')
if is_valid(assigned(Leanda, 2, _) and assigned(Olaf, 2, _)): print('(E)')
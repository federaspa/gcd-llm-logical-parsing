from z3 import *

## Declarations
divers_sort, (larue, ohba, pei, trevino, weiss, zacny) = EnumSort('divers', ['larue', 'ohba', 'pei', 'trevino', 'weiss', 'zacny'])
divers = [larue, ohba, pei, trevino, weiss, zacny]
dives = Function('dives', divers_sort, IntSort())

## Constraints

pre_conditions = []
d = Const('d', divers_sort)
pre_conditions.append(ForAll([d], And(1 <= dives(d), dives(d) <= 6)))
pre_conditions.append(Distinct([dives(d) for d in divers]))
pre_conditions.append(dives(trevino) < dives(weiss))
pre_conditions.append(Or(dives(larue) == 1, dives(larue) == 6))
pre_conditions.append(Not(dives(weiss) == 6))
pre_conditions.append(Not(dives(zacny) == 6))
pre_conditions.append(Implies(dives(pei) > dives(ohba), dives(pei) > dives(larue)))
d = Const('d', divers_sort)
pre_conditions.append(ForAll([d], And(1 <= dives(d), dives(d) <= 6)))
pre_conditions.append(Distinct([dives(d) for d in divers]))
pre_conditions.append(dives(trevino) < dives(weiss))
pre_conditions.append(Or(dives(larue) == 1, dives(larue) == 6))
pre_conditions.append(Not(dives(weiss) == 6))
pre_conditions.append(Not(dives(zacny) == 6))
pre_conditions.append(Implies(dives(pei) > dives(ohba), dives(pei) > dives(larue)))

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

if is_exception(dives(ohba), 3): print('(A)')
if is_exception(dives(weiss), 3): print('(B)')
if is_exception(dives(zacny), 3): print('(C)')
if is_exception(dives(pei), 4): print('(D)')
if is_exception(dives(weiss), 4): print('(E)')

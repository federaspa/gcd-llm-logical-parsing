from z3 import *

## Declarations
neighborhoods_sort, (hiddenhills, lakeville, nottingham, oldtown, parkplaza, sunnyside) = EnumSort('neighborhoods', ['hiddenhills', 'lakeville', 'nottingham', 'oldtown', 'parkplaza', 'sunnyside'])
days_sort, (monday, tuesday, wednesday, thursday, friday) = EnumSort('days', ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'])
neighborhoods = [hiddenhills, lakeville, nottingham, oldtown, parkplaza, sunnyside]
days = [monday, tuesday, wednesday, thursday, friday]
visited = Function('visited', days_sort, neighborhoods_sort, BoolSort())

## Constraints

pre_conditions = []
n = Const('n', neighborhoods_sort)
pre_conditions.append(ForAll([n], Sum([visited(d, n) for d in days])) == 1)
d = Const('d', days_sort)
pre_conditions.append(Exists([d], visited(d, hiddenhills)))
pre_conditions.append(Not(visited(friday, hiddenhills)))
pre_conditions.append(Implies(visited(d, oldtown), visited(d, hiddenhills) == 1))
pre_conditions.append(Implies(visited(wednesday, lakeville), visited(wednesday, lakeville)))
pre_conditions.append(And(visited(d, nottingham), visited(d, sunnyside)))
pre_conditions.append(Not(visited(d, nottingham) == visited(d, sunnyside)))
n = Const('n', neighborhoods_sort)
pre_conditions.append(ForAll([n], Sum([visited(d, n) for d in days])) == 1)
d = Const('d', days_sort)
pre_conditions.append(Exists([d], visited(d, hiddenhills)))
pre_conditions.append(Not(visited(friday, hiddenhills)))
pre_conditions.append(Implies(visited(d, oldtown), visited(d, hiddenhills) == 1))
pre_conditions.append(Implies(visited(wednesday, lakeville), visited(wednesday, lakeville)))
pre_conditions.append(And(visited(d, nottingham), visited(d, sunnyside)))
pre_conditions.append(Not(visited(d, nottingham) == visited(d, sunnyside)))

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

if is_valid(And(visited(monday, nottingham), visited(tuesday, lakeville), visited(wednesday, oldtown), visited(thursday, hiddenhills), visited(friday, sunnyside))): print('(A)')
if is_valid(And(visited(monday, nottingham), visited(tuesday, oldtown), visited(wednesday, hiddenhills), visited(thursday, sunnyside), visited(friday, parkplaza))): print('(B)')
if is_valid(And(visited(monday, oldtown), visited(tuesday, hiddenhills), visited(wednesday, lakeville), visited(thursday, nottingham), visited(friday, sunnyside))): print('(C)')
if is_valid(And(visited(monday, sunnyside), visited(tuesday, oldtown), visited(wednesday, lakeville), visited(thursday, hiddenhills), visited(friday, nottingham))): print('(D)')
if is_valid(And(visited(monday, sunnyside), visited(tuesday, parkplaza), visited(wednesday, nottingham), visited(thursday, oldtown), visited(friday, hiddenhills))): print('(E)')
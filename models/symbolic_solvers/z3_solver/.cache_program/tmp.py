from z3 import *

vehicles_sort, (hatchback, limousine, pickup, roadster, sedan, van) = EnumSort('vehicles', ['hatchback', 'limousine', 'pickup', 'roadster', 'sedan', 'van'])
days_sort = IntSort()
Monday, Tuesday, Wednesday, Thursday, Friday, Saturday = Ints('Monday Tuesday Wednesday Thursday Friday Saturday')
days = [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday]
vehicles = [hatchback, limousine, pickup, roadster, sedan, van]
serviced = Function('serviced', vehicles_sort, days_sort)

pre_conditions = []
pre_conditions.append(And(Monday == 1, Tuesday == 2, Wednesday == 3, Thursday == 4, Friday == 5))
v = Const('v', vehicles_sort)
pre_conditions.append(Exists([v], serviced(v) > serviced(hatchback)))
pre_conditions.append(And(serviced(roadster) > serviced(van), serviced(roadster) < serviced(hatchback)))
pre_conditions.append(Xor(Abs(serviced(pickup) - serviced(van)) == 1, Abs(serviced(pickup) - serviced(sedan)) == 1))
pre_conditions.append(Xor(serviced(sedan) < serviced(pickup), serviced(sedan) < serviced(limousine)))
pre_conditions.append(And(Monday == 1, Tuesday == 2, Wednesday == 3, Thursday == 4, Friday == 5))
v = Const('v', vehicles_sort)
pre_conditions.append(Exists([v], serviced(v) > serviced(hatchback)))
pre_conditions.append(And(serviced(roadster) > serviced(van), serviced(roadster) < serviced(hatchback)))
pre_conditions.append(Xor(Abs(serviced(pickup) - serviced(van)) == 1, Abs(serviced(pickup) - serviced(sedan)) == 1))
pre_conditions.append(Xor(serviced(sedan) < serviced(pickup), serviced(sedan) < serviced(limousine)))

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


if is_sat(And(serviced(pickup) == Tuesday, serviced(hatchback) == Wednesday, serviced(limousine) == Friday)): print('(A)')
if is_sat(And(serviced(pickup) == Tuesday, serviced(roadster) == Wednesday, serviced(hatchback) == Friday)): print('(B)')
if is_sat(And(serviced(sedan) == Tuesday, serviced(limousine) == Wednesday, serviced(hatchback) == Friday)): print('(C)')
if is_sat(And(serviced(van) == Tuesday, serviced(limousine) == Wednesday, serviced(hatchback) == Friday)): print('(D)')
if is_sat(And(serviced(van) == Tuesday, serviced(roadster) == Wednesday, serviced(limousine) == Friday)): print('(E)')
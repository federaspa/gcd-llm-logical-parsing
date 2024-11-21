from z3 import *

## Declarations
technicians_sort, (Stacy, Urma, Wim, Xena, Yolanda, Zane) = EnumSort('technicians', ['Stacy', 'Urma', 'Wim', 'Xena', 'Yolanda', 'Zane'])
machines_sort, (radios, televisions, VCRs) = EnumSort('machines', ['radios', 'televisions', 'VCRs'])
technicians = [Stacy, Urma, Wim, Xena, Yolanda, Zane]
machines = [radios, televisions, VCRs]
repair = Function('repair', technicians_sort, machines_sort, BoolSort())

## Constraints

pre_conditions = []
t1 = Const('t1', machines_sort)
t2 = Const('t2', machines_sort)
pre_conditions.append(ForAll([t1, t2], (repair(t1, t2) && repair(t2, t1) && t1 != t2)))
t1 = Const('t1', machines_sort)
t2 = Const('t2', machines_sort)
pre_conditions.append(is_valid(Exists([t1, t2], (repair(t1, t2) && repair(t2, t1) && t1 != t2))))
pre_conditions.append(options)
t1 = Const('t1', machines_sort)
t2 = Const('t2', machines_sort)
pre_conditions.append(ForAll([t1, t2], (repair(t1, t2) && repair(t2, t1) && t1 != t2)))
t1 = Const('t1', machines_sort)
t2 = Const('t2', machines_sort)
pre_conditions.append(is_valid(Exists([t1, t2], (repair(t1, t2) && repair(t2, t1) && t1 != t2))))
pre_conditions.append(options)

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

t1 = Const('t1', machines_sort)
t2 = Const('t2', machines_sort)
if is_valid(Exists([t1, t2], (repair(t1, t2) && repair(t2, t1) && t1 != t2))): print('(A)')
t1 = Const('t1', machines_sort)
t2 = Const('t2', machines_sort)
if is_valid(Exists([t1, t2], (repair(t1, t2) && repair(t2, t1) && t1 != t2))): print('(B)')
t1 = Const('t1', machines_sort)
t2 = Const('t2', machines_sort)
if is_valid(Exists([t1, t2], (repair(t1, t2) && repair(t2, t1) && t1 != t2))): print('(C)')
t1 = Const('t1', machines_sort)
t2 = Const('t2', machines_sort)
if is_valid(Exists([t1, t2], (repair(t1, t2) && repair(t2, t1) && t1 != t2))): print('(D)')
t1 = Const('t1', machines_sort)
t2 = Const('t2', machines_sort)
if is_valid(Exists([t1, t2], (repair(t1, t2) && repair(t2, t1) && t1 != t2))): print('(E)')

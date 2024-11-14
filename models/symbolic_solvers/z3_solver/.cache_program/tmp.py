from z3 import *

## Declarations
technicians_sort, (Stacy, Urma, Wim, Xena, Yolanda, Zane) = EnumSort('technicians', ['Stacy', 'Urma', 'Wim', 'Xena', 'Yolanda', 'Zane'])
machines_sort, (radios, televisions, VCRs) = EnumSort('machines', ['radios', 'televisions', 'VCRs'])
technicians = [Stacy, Urma, Wim, Xena, Yolanda, Zane]
machines = [radios, televisions, VCRs]
repairs = Function('repairs', technicians_sort, machines_sort, BoolSort())

## Constraints

pre_conditions = []
t = Const('t', technicians_sort)
pre_conditions.append(ForAll([t], Sum([repairs(t, m) for m in machines]) >= 1))
pre_conditions.append(And(repairs(Xena, radios), Sum([And(t != Xena, repairs(t, radios)) for t in technicians]) == 3))
pre_conditions.append(And(repairs(Yolanda, televisions), repairs(Yolanda, VCRs)))
m = Const('m', machines_sort)
pre_conditions.append(ForAll([m], Implies(repairs(Yolanda, m), Not(repairs(Stacy, m)))))
pre_conditions.append(Sum([repairs(Zane, m) for m in machines]) > Sum([repairs(Yolanda, m) for m in machines]))
m = Const('m', machines_sort)
pre_conditions.append(ForAll([m], Implies(repairs(Stacy, m), Not(repairs(Wim, m)))))
pre_conditions.append(Sum([repairs(Urma, m) for m in machines]) == 2)
t = Const('t', technicians_sort)
pre_conditions.append(ForAll([t], Sum([repairs(t, m) for m in machines]) >= 1))
pre_conditions.append(And(repairs(Xena, radios), Sum([And(t != Xena, repairs(t, radios)) for t in technicians]) == 3))
pre_conditions.append(And(repairs(Yolanda, televisions), repairs(Yolanda, VCRs)))
m = Const('m', machines_sort)
pre_conditions.append(ForAll([m], Implies(repairs(Yolanda, m), Not(repairs(Stacy, m)))))
pre_conditions.append(Sum([repairs(Zane, m) for m in machines]) > Sum([repairs(Yolanda, m) for m in machines]))
m = Const('m', machines_sort)
pre_conditions.append(ForAll([m], Implies(repairs(Stacy, m), Not(repairs(Wim, m)))))
pre_conditions.append(Sum([repairs(Urma, m) for m in machines]) == 2)

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

m = Const('m', machines_sort)
if is_sat(ForAll([m], repairs(Stacy, m) == repairs(Urma, m))): print('(A)')
m = Const('m', machines_sort)
if is_sat(ForAll([m], repairs(Urma, m) == repairs(Yolanda, m))): print('(B)')
m = Const('m', machines_sort)
if is_sat(ForAll([m], repairs(Urma, m) == repairs(Xena, m))): print('(C)')
m = Const('m', machines_sort)
if is_sat(ForAll([m], repairs(Wim, m) == repairs(Xena, m))): print('(D)')
m = Const('m', machines_sort)
if is_sat(ForAll([m], repairs(Xena, m) == repairs(Yolanda, m))): print('(E)')
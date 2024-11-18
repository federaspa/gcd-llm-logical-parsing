from z3 import *

## Declarations
children_sort, (fred, juan, marc, paul, nita, rachel, trisha) = EnumSort('children', ['fred', 'juan', 'marc', 'paul', 'nita', 'rachel', 'trisha'])
lockers_sort, (l1, l2, l3, l4, l5) = EnumSort('lockers', ['l1', 'l2', 'l3', 'l4', 'l5'])
children = [fred, juan, marc, paul, nita, rachel, trisha]
lockers = [l1, l2, l3, l4, l5]
assigned = Function('assigned', children_sort, lockers_sort, BoolSort())

## Constraints

pre_conditions = []
c = Const('c', children_sort)
pre_conditions.append(ForAll([c], Sum([assigned(c, l) for l in lockers])) == 1)
l = Const('l', lockers_sort)
pre_conditions.append(ForAll([l], Sum([assigned(c, l) for c in children])))
l1 = Const('l1', lockers_sort)
l2 = Const('l2', lockers_sort)
pre_conditions.append(ForAll([l1, l2], Implies(l1 != l2, assigned(c, l1) != assigned(c, l2))))
c = Const('c', children_sort)
pre_conditions.append(ForAll([c], Sum([assigned(c, l) for l in lockers])))
pre_conditions.append(assigned(juan, l) == True)
pre_conditions.append(assigned(rachel, l) == False)
pre_conditions.append(assigned(nita, l) == False)
pre_conditions.append(assigned(trisha, l) == False)
pre_conditions.append(assigned(fred, l3) == True)
c = Const('c', children_sort)
pre_conditions.append(ForAll([c], Sum([assigned(c, l) for l in lockers])) == 1)
l = Const('l', lockers_sort)
pre_conditions.append(ForAll([l], Sum([assigned(c, l) for c in children])))
l1 = Const('l1', lockers_sort)
l2 = Const('l2', lockers_sort)
pre_conditions.append(ForAll([l1, l2], Implies(l1 != l2, assigned(c, l1) != assigned(c, l2))))
c = Const('c', children_sort)
pre_conditions.append(ForAll([c], Sum([assigned(c, l) for l in lockers])))
pre_conditions.append(assigned(juan, l) == True)
pre_conditions.append(assigned(rachel, l) == False)
pre_conditions.append(assigned(nita, l) == False)
pre_conditions.append(assigned(trisha, l) == False)
pre_conditions.append(assigned(fred, l3) == True)

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

if is_accurate_list(assigned(juan, l)): print('(A)')
if is_accurate_list(assigned(juan, l)): print('(B)')
if is_accurate_list(assigned(juan, l)): print('(C)')
if is_accurate_list(assigned(juan, l)): print('(D)')
if is_accurate_list(assigned(juan, l)): print('(E)')

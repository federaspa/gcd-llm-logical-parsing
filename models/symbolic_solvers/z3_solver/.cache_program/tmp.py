from z3 import *

## Declarations
divisions_sort, (Operations, Production, Sales) = EnumSort('divisions', ['Operations', 'Production', 'Sales'])
days_sort = IntSort()
Monday, Tuesday, Wednesday, Thursday, Friday = Ints('Monday Tuesday Wednesday Thursday Friday')
days = [Monday, Tuesday, Wednesday, Thursday, Friday]
divisions = [Operations, Production, Sales]
toured = Function('toured', days_sort, divisions_sort)

## Constraints

pre_conditions = []
pre_conditions.append(And(Monday == 1, Tuesday == 2, Wednesday == 3, Thursday == 4, Friday == 5))
pre_conditions.append(Sum([toured(d) == Operations for d in days]) >= 1)
pre_conditions.append(Sum([toured(d) == Production for d in days]) >= 1)
pre_conditions.append(Sum([toured(d) == Sales for d in days]) >= 1)
pre_conditions.append(toured(Monday) != Operations)
pre_conditions.append(toured(Wednesday) != Production)
pre_conditions.append(And(Or(toured(Monday) == Sales, toured(Tuesday) == Sales), Or(toured(Tuesday) == Sales, toured(Wednesday) == Sales), Or(toured(Wednesday) == Sales, toured(Thursday) == Sales), Or(toured(Thursday) == Sales, toured(Friday) == Sales)))
pre_conditions.append(Sum([toured(d) == Sales for d in days]) == 2)
pre_conditions.append(Implies(toured(Thursday) == Operations, toured(Friday) == Production))
pre_conditions.append(toured(Monday) != toured(Tuesday))
pre_conditions.append(And(Monday == 1, Tuesday == 2, Wednesday == 3, Thursday == 4, Friday == 5))
pre_conditions.append(Sum([toured(d) == Operations for d in days]) >= 1)
pre_conditions.append(Sum([toured(d) == Production for d in days]) >= 1)
pre_conditions.append(Sum([toured(d) == Sales for d in days]) >= 1)
pre_conditions.append(toured(Monday) != Operations)
pre_conditions.append(toured(Wednesday) != Production)
pre_conditions.append(And(Or(toured(Monday) == Sales, toured(Tuesday) == Sales), Or(toured(Tuesday) == Sales, toured(Wednesday) == Sales), Or(toured(Wednesday) == Sales, toured(Thursday) == Sales), Or(toured(Thursday) == Sales, toured(Friday) == Sales)))
pre_conditions.append(Sum([toured(d) == Sales for d in days]) == 2)
pre_conditions.append(Implies(toured(Thursday) == Operations, toured(Friday) == Production))
pre_conditions.append(toured(Monday) != toured(Tuesday))

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

if is_exception(Or([And(toured(d) == Sales, And([Implies(d2 < d, toured(d2) != Production) for d2 in days])) for d in days])): print('(A)')
if is_sat(Or([And(toured(d) == Operations, And([Implies(d2 < d, toured(d2) != Production) for d2 in days])) for d in days])): print('(B)')
if is_sat(toured(Monday) == Sales): print('(C)')
if is_sat(toured(Tuesday) == Production): print('(D)')
if is_sat(toured(Wednesday) == Operations): print('(E)')
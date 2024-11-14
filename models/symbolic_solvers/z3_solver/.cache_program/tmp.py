from z3 import *

## Declarations
pieces_sort, (reciprocity, salammbo, trapezoid, vancouver, wisteria) = EnumSort('pieces', ['reciprocity', 'salammbo', 'trapezoid', 'vancouver', 'wisteria'])
pieces = [reciprocity, salammbo, trapezoid, vancouver, wisteria]
order = Function('order', pieces_sort, IntSort())

## Constraints

pre_conditions = []
p = Const('p', pieces_sort)
pre_conditions.append(ForAll([p], And(1 <= order(p), order(p) <= 5)))
pre_conditions.append(Distinct([order(p) for p in pieces]))
pre_conditions.append(order(salammbo) < order(vancouver))
pre_conditions.append(Or(And(order(trapezoid) < order(reciprocity), order(trapezoid) < order(salammbo)), And(order(trapezoid) > order(reciprocity), order(trapezoid) > order(salammbo))))
pre_conditions.append(Or(And(order(wisteria) < order(reciprocity), order(wisteria) < order(trapezoid)), And(order(wisteria) > order(reciprocity), order(wisteria) > order(trapezoid))))
p = Const('p', pieces_sort)
pre_conditions.append(ForAll([p], And(1 <= order(p), order(p) <= 5)))
pre_conditions.append(Distinct([order(p) for p in pieces]))
pre_conditions.append(order(salammbo) < order(vancouver))
pre_conditions.append(Or(And(order(trapezoid) < order(reciprocity), order(trapezoid) < order(salammbo)), And(order(trapezoid) > order(reciprocity), order(trapezoid) > order(salammbo))))
pre_conditions.append(Or(And(order(wisteria) < order(reciprocity), order(wisteria) < order(trapezoid)), And(order(wisteria) > order(reciprocity), order(wisteria) > order(trapezoid))))

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

if is_unsat(And(order(wisteria) == 1, order(trapezoid) == 3)): print('(A)')
if is_unsat(And(order(wisteria) == 1, order(vancouver) == 3)): print('(B)')
if is_unsat(And(order(wisteria) == 1, order(salammbo) == 4)): print('(C)')
if is_unsat(And(order(wisteria) == 1, order(vancouver) == 4)): print('(D)')
if is_unsat(And(order(wisteria) == 1, order(trapezoid) == 5)): print('(E)')
from z3 import *

computers_sort, (P, Q, R, S, T, U) = EnumSort('computers', ['P', 'Q', 'R', 'S', 'T', 'U'])
computers = [P, Q, R, S, T, U]
received = Function('received', computers_sort, BoolSort())
transmitted = Function('transmitted', computers_sort, computers_sort, BoolSort())
infected = Function('infected', computers_sort, BoolSort())

pre_conditions = []
pre_conditions.append(Sum([infected(c) for c in computers]) == 1)
c = Const('c', computers_sort)
pre_conditions.append(ForAll([c], Sum([transmitted(c, d) for d in computers]) <= 2))
pre_conditions.append(Sum([transmitted(S, d) for d in computers]) == 1)
pre_conditions.append(And(transmitted(d, R), transmitted(d, S)))
pre_conditions.append(Or(transmitted(R, Q), transmitted(T, Q)))
pre_conditions.append(Or(transmitted(T, P), transmitted(U, P)))
c = Const('c', computers_sort)
pre_conditions.append(ForAll([c], Sum([received(c, d) for d in computers]) == 1))
pre_conditions.append(infected(S))
pre_conditions.append(Sum([infected(c) for c in computers]) == 1)
c = Const('c', computers_sort)
pre_conditions.append(ForAll([c], Sum([transmitted(c, d) for d in computers]) <= 2))
pre_conditions.append(Sum([transmitted(S, d) for d in computers]) == 1)
pre_conditions.append(And(transmitted(d, R), transmitted(d, S)))
pre_conditions.append(Or(transmitted(R, Q), transmitted(T, Q)))
pre_conditions.append(Or(transmitted(T, P), transmitted(U, P)))
c = Const('c', computers_sort)
pre_conditions.append(ForAll([c], Sum([received(c, d) for d in computers]) == 1))
pre_conditions.append(infected(S))

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


if is_sat(transmitted(S, T)): print('(A)')
if is_sat(transmitted(T, P)): print('(B)')
if is_sat(Not(transmitted(Q, _))): print('(C)')
if is_sat(Not(transmitted(R, _))): print('(D)')
if is_sat(Not(transmitted(U, _))): print('(E)')
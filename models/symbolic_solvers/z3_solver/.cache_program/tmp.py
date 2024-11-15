from z3 import *

## Declarations
dishes_sort, (dish1, dish2, dish3, dish4, dish5, dish6) = EnumSort('dishes', ['dish1', 'dish2', 'dish3', 'dish4', 'dish5', 'dish6'])
shelves_sort, (bottom, middle, top) = EnumSort('shelves', ['bottom', 'middle', 'top'])
dishes = [dish1, dish2, dish3, dish4, dish5, dish6]
shelves = [bottom, middle, top]
placed = Function('placed', dishes_sort, shelves_sort, BoolSort())

## Constraints

pre_conditions = []
s = Const('s', shelves_sort)
pre_conditions.append(ForAll([s], Sum([placed(d, s) for d in dishes])))
pre_conditions.append(Sum([placed(d, bottom) for d in dishes]) <= 3)
pre_conditions.append(Sum([placed(d, middle) for d in dishes]) <= 3)
pre_conditions.append(Sum([placed(d, top) for d in dishes]) <= 3)
s = Const('s', shelves_sort)
pre_conditions.append(Exists([s], placed(dish2, s)))
s = Const('s', shelves_sort)
pre_conditions.append(Exists([s], placed(dish6, s)))
pre_conditions.append(And(placed(dish2, s1), placed(dish6, s2), s1 != s2, s1 != bottom, s2 != bottom))
pre_conditions.append(And(placed(dish6, s1), placed(dish5, s2), s1 != s2, s1 != bottom, s2 != bottom))
pre_conditions.append(Not(placed(dish1, s) == placed(dish4, s)))
s = Const('s', shelves_sort)
pre_conditions.append(ForAll([s], Sum([placed(d, s) for d in dishes])))
pre_conditions.append(Sum([placed(d, bottom) for d in dishes]) <= 3)
pre_conditions.append(Sum([placed(d, middle) for d in dishes]) <= 3)
pre_conditions.append(Sum([placed(d, top) for d in dishes]) <= 3)
s = Const('s', shelves_sort)
pre_conditions.append(Exists([s], placed(dish2, s)))
s = Const('s', shelves_sort)
pre_conditions.append(Exists([s], placed(dish6, s)))
pre_conditions.append(And(placed(dish2, s1), placed(dish6, s2), s1 != s2, s1 != bottom, s2 != bottom))
pre_conditions.append(And(placed(dish6, s1), placed(dish5, s2), s1 != s2, s1 != bottom, s2 != bottom))
pre_conditions.append(Not(placed(dish1, s) == placed(dish4, s)))

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

if is_valid(And(placed(dish1, bottom), placed(dish3, bottom), placed(dish6, middle), placed(dish2, top), placed(dish4, top), placed(dish5, top))): print('(A)')
if is_valid(And(placed(dish1, bottom), placed(dish3, bottom), placed(dish6, middle), placed(dish2, top), placed(dish4, top), placed(dish5, top))): print('(B)')
if is_valid(And(placed(dish2, bottom), placed(dish4, middle), placed(dish6, middle), placed(dish1, top), placed(dish3, top), placed(dish5, top))): print('(C)')
if is_valid(And(placed(dish3, bottom), placed(dish5, bottom), placed(dish6, middle), placed(dish1, top), placed(dish2, top), placed(dish4, top))): print('(D)')
if is_valid(And(placed(dish4, bottom), placed(dish6, bottom), placed(dish1, middle), placed(dish2, middle), placed(dish3, top), placed(dish5, top))): print('(E)')
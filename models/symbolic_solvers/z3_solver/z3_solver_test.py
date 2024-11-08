from .sat_problem_solver import LSAT_Z3_Program

if __name__=="__main__":
    completion = {
  "declarations": [
    "people = EnumSort([Vladimir, Wendy])",
    "meals = EnumSort([breakfast, lunch, dinner, snack])",
    "foods = EnumSort([fish, hot_cakes, macaroni, omelet, poached_eggs])",
    "eats = Function([people, meals] -> [foods])"
  ],
  "constraints": [
    "ForAll([m:meals], eats(Vladimir, m) != eats(Wendy, m))",
    "ForAll([p:people, f:foods], Count([m:meals], eats(p, m) == f) <= 1)",
    "ForAll([p:people], Or(eats(p, breakfast) == hot_cakes, eats(p, breakfast) == poached_eggs, eats(p, breakfast) == omelet))",
    "ForAll([p:people], Or(eats(p, lunch) == fish, eats(p, lunch) == hot_cakes, eats(p, lunch) == macaroni, eats(p, lunch) == omelet))",
    "ForAll([p:people], Or(eats(p, dinner) == fish, eats(p, dinner) == hot_cakes, eats(p, dinner) == macaroni, eats(p, dinner) == omelet))",
    "ForAll([p:people], Or(eats(p, snack) == fish, eats(p, snack) == omelet))",
    "eats(Wendy, lunch) == omelet"
  ],
  "options": [
    "is_valid(Exists([m:meals], eats(Vladimir, m) == fish))",
    "is_valid(Exists([m:meals], eats(Vladimir, m) == hot_cakes))",
    "is_valid(Exists([m:meals], eats(Vladimir, m) == macaroni))",
    "is_valid(Exists([m:meals], eats(Vladimir, m) == omelet))",
    "is_valid(Exists([m:meals], eats(Vladimir, m) == poached_eggs))"
  ]
}


    completion2 = {
  "declarations": [
    "technicians = EnumSort([Stacy, Urma, Wim, Xena, Yolanda, Zane])",
    "machines = EnumSort([radios, televisions, VCRs])",
    "repairs = Function([technicians, machines] -> [bool])"
  ],
  "constraints": [
    "ForAll([t:technicians], Count([m:machines], repairs(t, m)) >= 1)",
    "And(repairs(Xena, radios), Count([t:technicians], And(t != Xena, repairs(t, radios))) == 3)",
    "And(repairs(Yolanda, televisions), repairs(Yolanda, VCRs))",
    "ForAll([m:machines], Implies(repairs(Yolanda, m), Not(repairs(Stacy, m))))",
    "Count([m:machines], repairs(Zane, m)) > Count([m:machines], repairs(Yolanda, m))",
    "ForAll([m:machines], Implies(repairs(Stacy, m), Not(repairs(Wim, m))))",
    "Count([m:machines], repairs(Urma, m)) == 2"
  ],
  "options": [
    "is_sat(ForAll([m:machines], repairs(Stacy, m) == repairs(Urma, m)))",
    "is_sat(ForAll([m:machines], repairs(Urma, m) == repairs(Yolanda, m)))",
    "is_sat(ForAll([m:machines], repairs(Urma, m) == repairs(Xena, m)))",
    "is_sat(ForAll([m:machines], repairs(Wim, m) == repairs(Xena, m)))",
    "is_sat(ForAll([m:machines], repairs(Xena, m) == repairs(Yolanda, m)))"
  ]
}


    completion3 = {
  "declarations": [
    "CD_types = EnumSort([jazz, opera, pop, rap, soul])",
    "new_used = EnumSort([new, used])",
    "on_sale = Function([CD_types, new_used] -> [bool])"
  ],
  "constraints": [
    "And(on_sale(pop, used), Not(on_sale(opera, new)))",
    "Implies(And(on_sale(pop, new), on_sale(pop, used)), And(on_sale(soul, new), on_sale(soul, used)))",
    "Implies(And(on_sale(jazz, new), on_sale(jazz, used)), ForAll([nu:new_used], Not(on_sale(rap, nu))))",
    "Implies(And(Not(on_sale(jazz, new)), Not(on_sale(jazz, used))), on_sale(pop, new))",
    "Implies(Or(on_sale(rap, new), on_sale(rap, used)), ForAll([nu:new_used], Not(on_sale(soul, nu))))",
    "And(Count([t:CD_types], on_sale(t, used)) == 4, Count([t:CD_types], on_sale(t, new)) == 0)"
  ],
  "options": [
    "is_sat(Not(on_sale(jazz, used)))",
    "is_sat(Not(on_sale(opera, used)))",
    "is_sat(Not(on_sale(rap, used)))",
    "is_sat(And(Not(on_sale(jazz, new)), Not(on_sale(jazz, used))))",
    "is_sat(And(Not(on_sale(rap, new)), Not(on_sale(rap, used)), Not(on_sale(soul, new)), Not(on_sale(soul, used))))"
  ]
}


    completion4 = {
  "declarations": [
    "children = EnumSort([Fred, Juan, Marc, Paul, Nita, Rachel, Trisha])",
    "lockers = EnumSort([1, 2, 3, 4, 5])",
    "genders = EnumSort([boy, girl])",
    "assigned = Function([children] -> [lockers])",
    "gender = Function([children] -> [genders])"
  ],
  "constraints": [
    "ForAll([l:lockers], Or(Exists([c:children], assigned(c) == l), Exists([c1:children, c2:children], And(c1 != c2, And(assigned(c1) == l, assigned(c2) == l, gender(c1) != gender(c2))))))",
    "Exists([c:children, l:lockers], And(assigned(Juan) == l, assigned(c) == l, c != Juan))",
    "ForAll([l:lockers], Or(assigned(Rachel) != l, Not(Exists([c:children], And(c != Rachel, assigned(c) == l)))))",
    "ForAll([l:lockers], Implies(assigned(Nita) == l, And(assigned(Trisha) != l - 1, assigned(Trisha) != l + 1)))",
    "assigned(Fred) == 3",
    "And(assigned(Trisha) == 3, assigned(Marc) == 1, ForAll([c:children], Implies(c != Marc, assigned(c) != 1)))"
  ],
  "options": [
    "is_valid(Exists([c:children], And(assigned(Juan) == 4, assigned(c) == 4, c != Juan)))",
    "is_valid(Exists([c:children], And(assigned(Juan) == 5, assigned(c) == 5, c != Juan)))",
    "is_valid(assigned(Paul) == 2)",
    "is_valid(assigned(Rachel) == 2)",
    "is_valid(assigned(Rachel) == 5)"
  ]
}


    completion5 = {
  "declarations": [
    "vehicles = EnumSort([hatchback, limousine, pickup, roadster, sedan, van])",
    "days = EnumSort([1, 2, 3, 4, 5, 6])",
    "serviced = Function([vehicles] -> [days])"
  ],
  "constraints": [
    "ForAll([v:vehicles], And(1 <= serviced(v), serviced(v) <= 6))",
    "Exists([v:vehicles], serviced(v) > serviced(hatchback))",
    "And(serviced(roadster) > serviced(van), serviced(roadster) < serviced(hatchback))",
    "Xor(Abs(serviced(pickup) - serviced(van)) == 1, Abs(serviced(pickup) - serviced(sedan)) == 1)",
    "Xor(serviced(sedan) < serviced(pickup), serviced(sedan) < serviced(limousine))"
  ],
  "options": [
    "is_sat(And(serviced(pickup) == 2, serviced(hatchback) == 3, serviced(limousine) == 5))",
    "is_sat(And(serviced(pickup) == 2, serviced(roadster) == 3, serviced(hatchback) == 5))",
    "is_sat(And(serviced(sedan) == 2, serviced(limousine) == 3, serviced(hatchback) == 5))",
    "is_sat(And(serviced(van) == 2, serviced(limousine) == 3, serviced(hatchback) == 5))",
    "is_sat(And(serviced(van) == 2, serviced(roadster) == 3, serviced(limousine) == 5))"
  ]
}


    completion6 = {
  "declarations": [
    "vehicles = EnumSort([hatchback, limousine, pickup, roadster, sedan, van])",
    "days = IntSort([Monday, Tuesday, Wednesday, Thursday, Friday, Saturday])",
    "serviced = Function([vehicles] -> [days])"
  ],
  "constraints": [
    "ForAll([v:vehicles], And(1 <= serviced(v), serviced(v) <= 6))",
    "And(Monday == 1, Tuesday == 2, Wednesday == 3, Thursday == 4, Friday == 5)",
    "Exists([v:vehicles], serviced(v) > serviced(hatchback))",
    "And(serviced(roadster) > serviced(van), serviced(roadster) < serviced(hatchback))",
    "Xor(Abs(serviced(pickup) - serviced(van)) == 1, Abs(serviced(pickup) - serviced(sedan)) == 1)",
    "Xor(serviced(sedan) < serviced(pickup), serviced(sedan) < serviced(limousine))"
  ],
  "options": [
    "is_sat(And(serviced(pickup) == Tuesday, serviced(hatchback) == Wednesday, serviced(limousine) == Friday))",
    "is_sat(And(serviced(pickup) == Tuesday, serviced(roadster) == Wednesday, serviced(hatchback) == Friday))",
    "is_sat(And(serviced(sedan) == Tuesday, serviced(limousine) == Wednesday, serviced(hatchback) == Friday))",
    "is_sat(And(serviced(van) == Tuesday, serviced(limousine) == Wednesday, serviced(hatchback) == Friday))",
    "is_sat(And(serviced(van) == Tuesday, serviced(roadster) == Wednesday, serviced(limousine) == Friday))"
  ]
}

    completion7 = {
  "declarations": [
    "days = IntSort([Monday, Tuesday, Wednesday, Thursday, Friday])",
    "divisions = EnumSort([Operations, Production, Sales])",
    "toured = Function([days] -> [divisions])"
  ],
  "constraints": [
    "And(Monday == 1, Tuesday == 2, Wednesday == 3, Thursday == 4, Friday == 5)",
    "Count([d:days], toured(d) == Operations) >= 1",
    "Count([d:days], toured(d) == Production) >= 1",
    "Count([d:days], toured(d) == Sales) >= 1",
    "toured(Monday) != Operations",
    "toured(Wednesday) != Production",
    "And(Or(toured(Monday) == Sales, toured(Tuesday) == Sales), Or(toured(Tuesday) == Sales, toured(Wednesday) == Sales), Or(toured(Wednesday) == Sales, toured(Thursday) == Sales), Or(toured(Thursday) == Sales, toured(Friday) == Sales))",
    "Count([d:days], toured(d) == Sales) == 2",
    "Implies(toured(Thursday) == Operations, toured(Friday) == Production)",
    "toured(Monday) != toured(Tuesday)"
  ],
  "options": [
    "is_exception(Exists([d:days], And(toured(d) == Sales, ForAll([d2:days], Implies(d2 < d, toured(d2) != Production)))))",
    "is_sat(Exists([d:days], And(toured(d) == Operations, ForAll([d2:days], Implies(d2 < d, toured(d2) != Production)))))",
    "is_sat(toured(Monday) == Sales)",
    "is_sat(toured(Tuesday) == Production)",
    "is_sat(toured(Wednesday) == Operations)"
  ]
}

    z3_program = LSAT_Z3_Program(completion7)
    ans, error_message = z3_program.execute_program()
    # print(ans)
    print(error_message)
    print(z3_program.standard_code)
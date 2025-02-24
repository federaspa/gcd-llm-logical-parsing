# rules_fc.py

from pyke import contexts, pattern, fc_rule, knowledge_base

pyke_version = '1.1.1'
compiler_version = 1

def fact1(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Young', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Furry',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact2(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Nice', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        with knowledge_base.Gen_once if index == 1 \
                 else engine.lookup('facts', 'Furry', context,
                                    rule.foreach_patterns(1)) \
          as gen_1:
          for dummy in gen_1:
            engine.assert_('facts', 'Green',
                           (rule.pattern(0).as_data(context),
                            rule.pattern(1).as_data(context),)),
            rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact3(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Green', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Nice',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact4(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Nice', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        with knowledge_base.Gen_once if index == 1 \
                 else engine.lookup('facts', 'Green', context,
                                    rule.foreach_patterns(1)) \
          as gen_1:
          for dummy in gen_1:
            engine.assert_('facts', 'Big',
                           (rule.pattern(0).as_data(context),
                            rule.pattern(1).as_data(context),)),
            rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact5(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Green', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Smart',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact6(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Big', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        with knowledge_base.Gen_once if index == 1 \
                 else engine.lookup('facts', 'Young', context,
                                    rule.foreach_patterns(1)) \
          as gen_1:
          for dummy in gen_1:
            engine.assert_('facts', 'Round',
                           (rule.pattern(0).as_data(context),
                            rule.pattern(1).as_data(context),)),
            rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact7(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Green', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Big',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact8(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Young', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        engine.assert_('facts', 'Furry',
                       (rule.pattern(0).as_data(context),
                        rule.pattern(1).as_data(context),)),
        rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def fact9(rule, context = None, index = None):
  engine = rule.rule_base.engine
  if context is None: context = contexts.simple_context()
  try:
    with knowledge_base.Gen_once if index == 0 \
             else engine.lookup('facts', 'Furry', context,
                                rule.foreach_patterns(0)) \
      as gen_0:
      for dummy in gen_0:
        with knowledge_base.Gen_once if index == 1 \
                 else engine.lookup('facts', 'Smart', context,
                                    rule.foreach_patterns(1)) \
          as gen_1:
          for dummy in gen_1:
            engine.assert_('facts', 'Nice',
                           (rule.pattern(0).as_data(context),
                            rule.pattern(1).as_data(context),)),
            rule.rule_base.num_fc_rules_triggered += 1
  finally:
    context.done()

def populate(engine):
  This_rule_base = engine.get_create('rules')
  
  fc_rule.fc_rule('fact1', This_rule_base, fact1,
    (('facts', 'Young',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact2', This_rule_base, fact2,
    (('facts', 'Nice',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),
     ('facts', 'Furry',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact3', This_rule_base, fact3,
    (('facts', 'Green',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact4', This_rule_base, fact4,
    (('facts', 'Nice',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),
     ('facts', 'Green',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact5', This_rule_base, fact5,
    (('facts', 'Green',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact6', This_rule_base, fact6,
    (('facts', 'Big',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),
     ('facts', 'Young',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact7', This_rule_base, fact7,
    (('facts', 'Green',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact8', This_rule_base, fact8,
    (('facts', 'Young',
      (pattern.pattern_literal('Harry'),
       pattern.pattern_literal(True),),
      False),),
    (pattern.pattern_literal('Harry'),
     pattern.pattern_literal(True),))
  
  fc_rule.fc_rule('fact9', This_rule_base, fact9,
    (('facts', 'Furry',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),
     ('facts', 'Smart',
      (contexts.variable('x'),
       pattern.pattern_literal(True),),
      False),),
    (contexts.variable('x'),
     pattern.pattern_literal(True),))


Krb_filename = '../.cache_program/rules.krb'
Krb_lineno_map = (
    ((12, 16), (3, 3)),
    ((17, 19), (5, 5)),
    ((28, 32), (9, 9)),
    ((33, 37), (10, 10)),
    ((38, 40), (12, 12)),
    ((49, 53), (16, 16)),
    ((54, 56), (18, 18)),
    ((65, 69), (22, 22)),
    ((70, 74), (23, 23)),
    ((75, 77), (25, 25)),
    ((86, 90), (29, 29)),
    ((91, 93), (31, 31)),
    ((102, 106), (35, 35)),
    ((107, 111), (36, 36)),
    ((112, 114), (38, 38)),
    ((123, 127), (42, 42)),
    ((128, 130), (44, 44)),
    ((139, 143), (48, 48)),
    ((144, 146), (50, 50)),
    ((155, 159), (54, 54)),
    ((160, 164), (55, 55)),
    ((165, 167), (57, 57)),
)

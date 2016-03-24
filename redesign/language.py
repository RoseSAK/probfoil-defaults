from __future__ import print_function


from collections import defaultdict
from itertools import product

from problog.logic import Term, is_variable, Var, Clause, And
from problog.engine import DefaultEngine
from problog import get_evaluatable


class BaseLanguage(object):
    """Base class for languages."""

    def __init__(self):
        pass

    def refine(self, rule):
        """Generate one refinement for the given rule.

        :param rule: rule for which to generate refinements
        """
        raise NotImplementedError('abstract method')


class TypeModeLanguage(BaseLanguage):
    """Typed and Mode-based language."""

    MODE_EXIST = '+'     # use existing variable
    MODE_NEW = '-'       # create new variable (do not reuse existing)
    MODE_CONSTANT = 'c'  # insert constant

    def __init__(self):
        BaseLanguage.__init__(self)

        self._types = {}    # dict[tuple[str,int],tuple[str]]: signature / argument types
        self._values = defaultdict(set)   # dict[str, set[Term]]: values in data for given type
        self._modes = []    # list[tuple] : list of functor, modestr pairs

    def add_types(self, functor, argtypes):
        """Add type information for a predicate.

        Type information has to be unique for a functor/arity combination.

        :param functor: functor of the predicate
        :type functor: str
        :param argtypes: types of the arguments (arity is length of this list)
        :type argtypes: list[str]
        :raise ValueError: duplicate type definition is given
        """
        key = (functor, len(argtypes))
        if key in self._types:
            raise ValueError("A type definition already exists for '%s/%s'."
                             % (functor, len(argtypes)))
        else:
            self._types[key] = argtypes

    def add_modes(self, functor, argmodes):
        """Add mode information for a predicate.

        :param functor: functor of the predicate
        :type functor: str
        :param argmodes: modes of the arguments (arity is length of this list)
        :type argmodes: list[str]
        """
        self._modes.append((functor, argmodes))

    def add_values(self, typename, *values):
        """Add a value for a predicate.

        :param typename: name of the type
        :type typename: str
        :param values: value to add (can be multiple)
        :type values: collections.Iterable[Term]
        """
        for value in values:
            self._values[typename].add(value)

    def refine(self, rule):
        """Generate one refinement for the given rule.

        :param rule: rule for which to generate refinements
        :return: generator of literals that can be added to the rule
        :rtype: collections.Iterable[Term]
        """
        varcount = len(rule.get_variables())
        variables = self.get_variable_types(*rule.get_literals())

        if rule.get_literal():
            generated = rule.get_literal().refine_state
        else:
            generated = set()

        # 1) a positive refinement
        for functor, modestr in self._modes:
            arguments = []
            arity = len(modestr)
            types = self.get_argument_types(functor, arity)
            for argmode, argtype in zip(modestr, types):
                if argmode == '+':
                    # All possible variables of the given type
                    arguments.append(variables.get(argtype, []))
                elif argmode == '-':
                    # A new variable
                    arguments.append([Var('#')])  # what about adding a term a(X,X) where X is new?
                elif argmode == 'c':
                    # Add a constant
                    arguments.append(self.get_type_values(argtype))
                    pass
                else:
                    raise ValueError("Unknown mode specifier '%s'" % argmode)
            for args in product(*arguments):
                t = Term(functor, *args)
                if t not in generated:
                    generated.add(t)
                    t = t.apply(TypeModeLanguage._ReplaceNew(varcount))
                    t.refine_state = generated.copy()
                    yield t

        # 2) a negative refinement
        for functor, modestr in self._modes:
            if '-' in modestr:
                # No new variables allowed for negative literals
                continue
            arguments = []
            arity = len(modestr)
            types = self.get_argument_types(functor, arity)
            for argmode, argtype in zip(modestr, types):
                if argmode == '+':
                    # All possible variables of the given type
                    arguments.append(variables.get(argtype, []))
                elif argmode == 'c':
                    # Add a constant
                    arguments.append(self.get_type_values(argtype))
                    pass
                else:
                    raise ValueError("Unknown mode specifier '%s'" % argmode)
            for args in product(*arguments):
                t = -Term(functor, *args)
                if t not in generated:
                    generated.add(t)
                    t = t.apply(TypeModeLanguage._ReplaceNew(varcount))
                    t.refine_state = generated.copy()
                    yield t

    def get_type_values(self, typename):
        """Get all values that occur in the data for a given type.

        :param typename: name of type
        :type typename: str
        :return: set of values
        :rtype: set[Term]
        """
        return self._values.get(typename, [])

    def get_argument_types(self, functor, arity):
        """Get the types of the arguments of the given predicate.

        :param functor: functor of the predicate
        :type functor: str
        :param arity: arity of the predicate
        :type arity: int
        :return: tuple of type descriptors, one for each argument
        :rtype: tuple[str]
        """
        return self._types[(functor, arity)]

    def get_variable_types(self, *literals):
        """Get the types of all variables that occur in the given literals.

        :param literals: literals to extract variables from
        :type literals: collections.Iterable[Term]
        :return: dictionary with list of variables for each type
        :rtype: dict[str, list[Term]]
        """
        result = defaultdict(list)
        for lit in literals:
            if lit.is_negated():
                lit = -lit
            types = self.get_argument_types(lit.functor, lit.arity)
            for arg, argtype in zip(lit.args, types):
                if is_variable(arg) or arg.is_var():
                    result[argtype].append(arg)
        return result

    def load(self, data):
        """Load from data.

        :param data: datafile
        :type data: DataFile
        """

        for typeinfo in data.query('base', 1):
            typeinfo = typeinfo[0]
            self.add_types(typeinfo.functor, list(map(str, typeinfo.args)))

        for modeinfo in data.query('mode', 1):
            modeinfo = modeinfo[0]
            self.add_modes(modeinfo.functor, list(map(str, modeinfo.args)))

        for predicate, types in self._types.items():
            arg_values = zip(*data.query(*predicate))
            for a, t in zip(arg_values, types):
                self.add_values(t, *a)

    class _ReplaceNew(object):
        """Helper class for replacing new variables (indicated by name '#') by unique variables.

        :param count: the current number of variables
        :type count: int
        """

        def __init__(self, count):
            self.count = count

        def _get_name(self, index):
            """Get a name for the new variable.
            This name will be a single letter from A to Z, or V<num> if index > 25.

            :param index: number of the variable
            :type index: int
            :return: name of variable
            :rtype: str
            """
            if index < 26:
                return chr(65 + index)
            else:
                return 'V%d' % index

        def __getitem__(self, name):
            if name == '#':
                name = self._get_name(self.count)
                self.count += 1
                return Var(name)
            else:
                return Var(name)




# class Language(object):
#
#     def __init__(self) :
#         self.__types = {}
#         self.__values = defaultdict(set)
#         self.__modes = {}
#         self.__targets = []
#         self.__varcount = 0
#         self.learning_problem = None
#
#     modes = property( lambda s : s.__modes )
#
#     def initialize(self, knowledge):
#         predicates = list(self.__modes) + self.__targets
#         for predicate, arity in predicates:
#             # Load types
#             types = knowledge.base(predicate, arity)
#             if len(types) == 1:
#                 types = types[0]
#                 self.setArgumentTypes(Term(predicate, *types))
#             elif len(types) > 1:
#                 raise Exception("Multiple 'base' declarations for predicate '%s/%s'!" % (predicate, arity))
#             else :
#                 self.setArgumentTypes(Term(predicate, *([Term('id')] * arity)))
#                 print ("Missing 'base' declaration for predicate '%s/%s'!" % (predicate, arity), file=sys.stderr)
#
#             values = list(zip(*knowledge.values(predicate, arity)))
#             for tp, vals in zip(types, values):
#                 self.__values[tp] |= set(vals)
#
#     def addTarget(self, predicate, arity) :
#         self.__targets.append( (predicate, arity) )
#
#     def addValue(self, typename, value) :
#         self.__values[typename].add(value)
#
#     def setArgumentTypes(self, literal) :
#         self.__types[(literal.functor, literal.arity)] = literal.args
#
#     def setArgumentModes(self, literal) :
#         self.__modes[(literal.functor, literal.arity)] = literal.args
#
#     def getArgumentTypes(self, literal=None, key=None) :
#         if literal:
#             key = (literal.functor, literal.arity)
#         return self.__types.get(key,[])
#
#     def getArgumentModes(self, literal=None, key=None) :
#         if literal:
#             key = (literal.functor, literal.arity)
#         return self.__modes.get(key,[])
#
#     def getTypeValues(self, typename):
#         return self.__values.get(typename, [])
#
#     def getArgumentValues(self, literal):
#         types = self.getArgumentTypes(literal)
#         return list( product( *[ self.getTypeValues(t) for t in types ] ) )
#
#     def getTypedVariables(self, literal):
#         types = self.getArgumentTypes(literal)
#         result = []
#         for vn, vt in zip(literal.args, types) :
#             if is_variable(vn) or vn.is_var():
#                 result.append((vn, vt))
#         return set(result)
#
#
#
#
#     def newVar(self) :
#         self.__varcount += 1
#         return Var('Var_' + str(self.__varcount))
#
#     def relationsByConstant(self, constant ) :
#         kb = self.learning_problem.knowledge
#
#         c,vt = constant
#
#         res = []
#         varnames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#
#         goals = []
#         for pred_id in self.__modes:
#             if pred_id[1] < 2:
#                 continue
#             index = [i for i, x in enumerate(self.getArgumentTypes(key=pred_id)) if x == vt]
#             for i in index:
#                 vars = [varnames[j] for j in range(0, pred_id[1])]
#                 args = vars[:]
#                 args[i] = c
#                 goals.append(Term(pred_id[0], *args))
#         return ( map( Literal.parse, kb.query_goals(goals) ) )
#
# #        print (res)
# #
# #         res = []
# #         res += [ Literal('parent',(c,r[0])) for r in kb.query( Literal('parent', (c,'Y')), 'Y' ) ]
# #         res += [ Literal('parent',(r[0],c)) for r in kb.query( Literal('parent', ('Y',c)), 'Y' ) ]
# #         res += [ Literal('married',(c,r[0])) for r in kb.query( Literal('married', (c,'Y')), 'Y' ) ]
# #         res += [ Literal('married',(r[0],c)) for r in kb.query( Literal('married', ('Y',c)), 'Y' ) ]
#         with Log('nodes', constant=c, extra=res) : pass
# #         print ('A',res)
#         return res
#
#     def rpf_eval_path(self, path, rule, prd ) :
#         #print ('EVALUATING:', path)
#
#         evaluation_process = 4
#
#         if evaluation_process == 1 :
#             predict = [0] * len(rule.examples)
#             qs = []
#             for i, ex in enumerate(rule.examples) :
#
#                 if rule.score_correct[i] - prd[i] > 0 :
#                     assign = dict( zip( rule.target.arguments, ex ) )
#                     qs.append( MultiLiteral(*path).withAssign(assign) )
#
#             j = 0
#             res = rule.knowledge.verify(qs)
#             for i, ex in enumerate(rule.examples) :
#                 if rule.score_correct[i] - prd[i] > 0 :
#                     if res[j] == '1' : predict[i] = 1
#                     j+=1
#
#             return predict
#
#         elif evaluation_process == 2 :
#
#
#             predict = [0] * len(rule.examples)
#             for i, ex in enumerate(rule.examples) :
#                 if prd[i] < 1 :
#                     assign = dict( zip( rule.target.arguments, ex ) )
#                     eval = rule.knowledge.query( MultiLiteral(*path).withAssign(assign), ('t') )
#                     if eval : predict[i] = 1
#             return predict
#
#         elif evaluation_process == 3 :
#             eval = (set(map(tuple, rule.knowledge.query( path, rule.target.arguments ))) )
#             print (len(eval))
#             predict = [0] * len(rule.examples)
#             for i, ex in enumerate(rule.examples) :
#                 if prd[i] < 1 :
#                     if ex in eval :
#                         predict[i] = 1
#             return predict
#
#         else :
#             # TODO only evaluate rule for remaining positive examples
#             new_rule = RuleHead(previous=rule) + MultiLiteral( *path)
#             s = new_rule.score  # Force computation
#             return new_rule.getScorePredict()
#
#     def rpf_init(self, rule, num_paths=0, pos_threshold=0.1, max_level=2) :
#
#      with Timer(category="rpf") :
#       with Log('rpf_init') :
#         R = RPF(rule)
#         self.RPF_paths = list(R.paths)
#
#
#     def refinements(self, variables, use_vars) :
#
#         existing_variables = defaultdict(list)
#         existing_variables_set = set([])
#         for varname, vartype in variables :
#             existing_variables[vartype].append( varname )
#             existing_variables_set.add(varname)
#
#         if use_vars is not None:
#             use_vars = set(use_vars)  # set( [ varname for varname, vartype in use_vars ] )
#
#         for pred_id in self.__modes:
#             pred_name = pred_id[0]
#             arg_info = list(zip(self.getArgumentTypes(key=pred_id), self.getArgumentModes(key=pred_id)))
#             for args in self._build_refine(existing_variables, True, arg_info, use_vars):
#                 new_lit = Term(pred_name, *args)
#                 if new_lit.variables() & existing_variables_set:
#                     yield new_lit
#
#             if not self.learning_problem.NO_NEGATION:
#                 for args in self._build_refine(existing_variables, False, arg_info, use_vars):
#                     new_lit = -Term(pred_name, *args)
#                     if new_lit.variables() & existing_variables_set:
#                         yield new_lit
#
#     def _build_refine_one(self, existing_variables, positive, arg_type, arg_mode):
#         arg_mode = str(arg_mode)
#         if arg_mode in '+-':
#             for var in existing_variables[arg_type] :
#                 yield var
#         if arg_mode == '-' and positive and not self.learning_problem.RPF:
#             yield '#'
#         if arg_mode == 'c':
#             if positive:
#                 for val in self.kb.types[arg_type]:
#                     yield val
#             else:
#                 yield '_'
#
#     def _build_refine(self, existing_variables, positive, arg_info, use_vars) :
#         if arg_info :
#             for arg0 in self._build_refine_one(existing_variables, positive, arg_info[0][0], arg_info[0][1]) :
#                 if use_vars != None and arg0 in use_vars :
#                     use_vars1 = None
#                 else :
#                     use_vars1 = use_vars
#                 for argN in self._build_refine(existing_variables, positive, arg_info[1:], use_vars1) :
#                     if arg0 == '#':
#                         yield [self.newVar()] + argN
#                     else :
#                         yield [arg0] + argN
#         else :
#             if use_vars == None :
#                 yield []
#
# import os
# def bin_path(relative) :
#     return os.path.join( os.path.split( os.path.abspath(__file__) )[0], relative )
#
# class RPF(object) :
#
#     def __init__(self, rule) :
#         self.rule = rule
#         self.paths = None
#         self.rule.knowledge._create_file_det()
#         self.evaluate(self.rule)
#
#     def evaluate(self, rule) :
#         assert(rule.target.arity == 2)
#
#         modes = str([ Literal( pred_id[0], ['_'] * pred_id[1]) for pred_id in rule.language.modes if pred_id[1] == 2 ])
#
#         threshold = 0.1         # discard example if it's score falls below this value
#         eval_cutoff = 20        # evaluate paths when less than this number of examples is removed => None means always evaluate
#         stop_threshold = 0.8    # stop when this percentage of positive weight is covered by the rules
#         maxl = 4
#
#         stop_threshold = sum(rule.score_correct) * (1.0 - stop_threshold)
#
#         examples = [ i for i, s in enumerate(rule.score_correct) if s > threshold ]
#         predicts = [ 0 ] * len(rule.examples)
#
#         paths = set([])
#         paths_to_eval = set([])
#         paths_to_discard = set([])
#
#         eval_rule = RuleHead(previous=rule)
#
#         while examples :
#
#             # 1) Order constants by total score of examples in which they occur
#             scores = defaultdict(float)
#             for i in examples :
#                 ex = rule.examples[i]
#                 s = rule.score_correct[i] - predicts[i]
#                 for j,e in enumerate(ex) :
#                     scores[(e,j)] += s
#             scores = list(sorted([ (s,) + e for e, s in scores.items() ]))
#
#             # 2) Select the best constant
#             score, constant, position = scores[-1]
#
#             # 3) Construct target
#             args = ['_'] * rule.target.arity
#             args[ position ] = constant
#             target = rule.target.with_args(*args)
#             varargs = rule.target.arguments[:]
#             v1 = varargs.pop( position )
#             varargs = [v1] + varargs
#
#             # 4) Construct list of constants
#             constants = []
#             remaining_examples = []
#             for i in examples :
#                 ex = rule.examples[i]
#                 if ex[position] == constant :
#                     constants.append( ex[1-position] )
#                 else :
#                     remaining_examples.append( i )
#             if rule.learning_problem.VERBOSE > 9 :
#                 print ('RPF:', target, '-', len(examples), len(remaining_examples))
#             with Log('step', target=target, before=len(examples), after=len(remaining_examples)) : pass
#
#             # 5) Find paths for the given constants
#             current_paths = self._call_yap( target, varargs, modes, constants, maxl=maxl )
#             # 6) Evaluate paths (probabilistically)
#             # Only evaluate on remaining examples
#             for path in current_paths :
#                 if not path in paths :
#                     if rule.learning_problem.VERBOSE > 9 :
#                         print ('PATH FOUND', path)
#                     with Log('found', path=path) : pass
#                     paths.add(path)
#                     paths_to_eval.add(path)
#
#             eval_rule.example_filter = remaining_examples
#             if eval_cutoff == None or len(examples) - len(remaining_examples) < eval_cutoff :
#                 for path in paths_to_eval :
#                     new_rule = eval_rule + MultiLiteral( *path)
#                     s = new_rule.score  # Force computation
#                     s = sum(new_rule.getScorePredict())
#                     if new_rule.invalid_scores : # Evaluation error
#                         paths_to_discard.add(path)
#                         if rule.learning_problem.VERBOSE > 9 :
#                             print ('PATH DISCARDED', path)
#
#
#                     else :
#                         for i in remaining_examples :
#                             predicts[i] = max( predicts[i], new_rule.getScorePredict(i) )
#                 paths_to_eval = set([])
#
#             examples = [ i for i in remaining_examples if rule.score_correct[i] - predicts[i] > threshold ]
#
#             remain_score = sum([ rule.score_correct[i] - predicts[i] for i in remaining_examples ])
#             if remain_score < stop_threshold :
#                 break
#
#         if rule.learning_problem.VERBOSE > 9 :
#             print (paths)
#         with Log('paths', paths='\n'.join( map(str,paths )), discard='\n'.join( map(str,paths_to_discard ))) : pass
#
#         self.paths = list(paths - paths_to_discard)
#
#
#     def _call_yap( self, target, args, modes, constants, maxl=-1) :
#
#         DATAFILE = self.rule.knowledge.datafile_det
#
#         RPF_PROLOG = bin_path('rpf.pl')
#         import subprocess
# #        try :
#         #print (' '.join(['yap', "-L", RPF_PROLOG , '--', DATAFILE, str(maxl), "'" + str(target) + "'", "'" + modes + "'" ] + constants))
#
#         output = subprocess.check_output(['yap', "-L", RPF_PROLOG , '--', DATAFILE, str(maxl), str(target), modes ] + constants )
#
# #         print ('#### OUTPUT #####')
# #         print (output)
# #         print ('#################')
#
#
#         paths = []
#         for line in output.strip().split('\n') :
#             parts = line.split('|')
#             varname = parts[0]
#             if len(parts) <= 1 or not '(' in parts[1] :
#                 continue
#             path = tuple(sorted(map( lambda x : Literal.parse(x).withAssign( { varname : args[1], 'V_0' : args[0] } ), parts[1:] ) ))
#             paths.append(path)
#         return paths
#
# #        except subprocess.CalledProcessError :
# #            print ('Error during rpf', file=sys.stderr)
# #            with Log('error', context='grounding') : pass
#
#
#! /usr/bin/env python3

from __future__ import print_function

from collections import namedtuple, defaultdict
from util import Timer, Log

import re, os

def is_var(arg) :
    return arg == None or arg[0] == '_' or (arg[0] >= 'A' and arg[0] <= 'Z')

def strip_negation(name) :
    if name.startswith('\+') :
        return name[2:].strip()
    else :
        return name

def true(f) :
    for x in f :
        return True
    return False

class Literal(object) :
    
    def __init__(self, kb, functor, args) :
        self.functor = functor
        self.args = args
        self.kb = kb
    
    identifier = property(lambda s : (s.functor, len(s.args) ) )
                
    def __str__(self) :
        return '%s(%s)' % (self.functor, ','.join(self.args))

    def __repr__(self) :
        return '%s(%s)' % (self.functor, ','.join(self.args))

    def __hash__(self) :
        return hash(str(self))

    def __eq__(self, other) :
        return str(self) == str(other)
        
    def __ne__(self, other) :
        return str(self) != str(other)
        
    def __lt__(self, other) :
        return str(self) < str(other)
        
    def __cmp__(self, other) :
        if self.isNegated() and not other.isNegated() :
            return -1
        elif other.isNegated() and not other.isNegated() :
            return 1
        else :
            return 0
        
    def isNegated(self) :
        return self.functor.startswith('\+')
        
    def unify(self, ground_args) :
        
        result = {}
        for ARG, arg in zip(self.args, ground_args) :
            if is_var(ARG) and not ARG == '_' :
                result[ARG] = arg
        return result  
        
    def assign(self, subst) :
        return Literal(self.kb, self.functor, [ subst.get(arg, arg) for arg in self.args ])
        
    def variables(self) :
        result = defaultdict(set)
        types = self.kb.argtypes(strip_negation(self.functor), len(self.args))
        for tp, arg in zip(types, self.args) :
            if is_var(arg) and not arg == '_' :
                result[tp].add(arg)
        return result
        
    @classmethod
    def parse(cls, kb, string) :
        regex = re.compile('\s*(?P<name>[^\(]+)\((?P<args>[^\)]+)\)[,.]?\s*')
        result = []
        for name, args in regex.findall(string) :
            result.append( Literal( kb, name , map(lambda s : s.strip(), args.split(','))) )
        return result
        
class FactDB(object) :
    
    def __init__(self, idtypes=[]) :        
        self.predicates = {}
        self.idtypes = idtypes
        self.__examples = None
        self.types = defaultdict(set)
        self.modes = {}
        self.learn = []
        self.newvar_count = 0
        self.probabilistic = set([])
        self.prolog_file = None
        
    def newVar(self) :
        self.newvar_count += 1
        return 'X_' + str(self.newvar_count)
            
    def register_predicate(self, name, args) :
        arity = len(args)
        identifier = (name, arity)
        
        if not identifier in self.predicates :
            index = [ defaultdict(set) for i in range(0,arity) ] 
            self.predicates[identifier] = (index, [], args, [])
            
    def is_probabilistic(self, literal) :
        return literal.identifier in self.probabilistic
            
    def add_mode(self, name, args) :
        arity = len(args)
        identifier = (name, arity)
        self.modes[identifier] = args
        
    def add_learn(self, name, args) :
        self.learn.append( (name, args) )
        
    def add_fact(self, name, args, p=1.0) :
        arity = len(args)
       # self.register_predicate(name, arity)
        identifier = (name, arity)
        index, values, types, probs = self.predicates[identifier]
        
        if p < 1.0 :
            self.probabilistic.add(identifier)
        
        arg_index = len(values)
        values.append( args )
        probs.append( p )

        for i, arg in enumerate(args) :
            self.types[types[i]].add(arg)
            index[i][arg].add(arg_index)
        #index[-1][tuple(arg)].add(arg_index)
            
    def argtypes(self, name, arity) :
        identifier = (name, arity)
        return self.predicates[identifier][2]
        
    def ground_fact(self, name, args) :
        if name.startswith('\+') :
            negated = True
            name = strip_negation(name)
        else :
            negated = False

        arity = len(args)
        identifier = (name, arity)
        
        index, values, types, probs = self.predicates[identifier]
        
        result = set(range(0, len(values))) 
        for i,arg in enumerate(args) :
            if not is_var(arg) :
                result &= index[i][arg]
            if not result : break        
                
        if negated :
            if len(result) == 0 :
                return [ args ]
            else :
                return []
        else :
            return [ values[i] for i in result ]

    def find_fact(self, name, args) :
        if name.startswith('\+') :
           # raise NotImplementedError("No support for negative literals yet!")
            negated = True
            name = strip_negation(name)
        else :
            negated = False

        arity = len(args)
        identifier = (name, arity)
        
        index, values, types, probs = self.predicates[identifier]
        
        result = set(range(0, len(values))) 
        for i,arg in enumerate(args) :
            if not is_var(arg) :
                result &= index[i][arg]
            if not result : break        
        
        probability = sum( probs[i] for i in result )        
        if negated :
            return 1.0 - probability
        else :
            return probability

    def __str__(self) :
        s = ''
        for name, arity in self.predicates :
            for args in self.predicates[(name,arity)][1] :
                s += '%s(%s)' % (name, ','.join(args)) + '.\n'
        return s
          
    def reset_examples(self) :          
        self.__examples = None
        
    def _get_examples(self) :
        if self.__examples == None :
            from itertools import product
            self.__examples = list(product(*[ self.types[t] for t in self.idtypes ]))
        return self.__examples
    
    examples = property(_get_examples)
        
    def __getitem__(self, index) :
        return self.examples[index]
        
    def __len__(self) :
        return len(self.examples)
        
    def __iter__(self) :
        return iter(range(0,len(self.examples)))
        
    def clause(self, head, body, args) :
        subst = head.unify( args )
        for sol in self.query(body, subst) :
            yield sol

    def query_single(self, literals, substitution) :
        return true(self.query(literals, substitution))

    def query(self, literals, substitution) :
        if not literals :  # reached end of query
            yield substitution
        else :
            head, tail = literals[0], literals[1:]
            head_ground = head.assign(substitution)     # assign known variables
            
            for match in self.ground_fact(head_ground.functor, head_ground.args) :
                # find match for head
                new_substitution = dict(substitution)
                new_substitution.update( head_ground.unify( match ) )
                for sol in self.query(tail, new_substitution) :
                    yield sol
                    
    def toPrologFile(self) :
        if self.prolog_file == None :
            self.prolog_file = '/tmp/pf.pl'
            with open(self.prolog_file, 'w') as f :
                for pred in self.predicates :
                    index, values, types, probs = self.predicates[pred]
                    for args in values :
                        print('%s(%s).' % (pred[0], ', '.join(args) ), file=f)
        return self.prolog_file                

class Rule(object) :
    
    def __init__(self, head, body=[], tmp=False) :
        self.head = head
        self.body = body
        
        if not tmp :
            self.variables, self.varnames = self.extract_variables([self.head] + self.body)
#        self.varcount = sum(map(len,self.variables))
        
    def evaluate(self, kb, example) :
        subst = self.head.unify( example )
        tv_body = kb.query_single(self.body, subst)
        return tv_body

    def evaluate_extensions_non_prob(self, kb, examples, extensions) :
        scores = [ [] for ext in extensions ]
        for example in examples :
            next_example = False
            subst = self.head.unify( example )
            tv_body = [ False ] * len(extensions)
            n_false = len(tv_body)
            for body_vars in kb.query(self.body, subst) :
                for lit_i, lit in enumerate(extensions) :
                    if tv_body[lit_i] == False :
                        lit_val = kb.query_single([lit],body_vars)
                        if lit_val :
                            tv_body[lit_i] = lit_val
                            n_false -= 1
                            if n_false == 0 :
                                next_example = True
                                break
                if next_example :
                    break
            for lit_i, lit in enumerate(extensions) :
                scores[lit_i].append(tv_body[lit_i])
        return zip(extensions, scores)
        
    def evaluate_extensions_prob(self, kb, examples, extensions) :
        if extensions :
            outfile = open('/tmp/pfprob.pl','w')
            
            # 1) Write out data
            datafile = kb.toPrologFile()
            print(":- consult('%s')." % (datafile), file=outfile)
            
            # 2) Write out current clause as Prolog clause
            old_head = Literal(kb, 'pf_current_rule', self.varnames)
            old_body = self.body
            old_rule = Rule(old_head, old_body, True)
            
            print(old_rule, file=outfile)
            
            new_rules = []
            queries = []
            for lit_id, ext in enumerate(extensions) :
                new_head = Literal(kb, 'pf_new_rule_' + str(lit_id), self.head.args)
                new_body = [ old_head, ext ]
                new_rule = Rule(new_head, new_body, True)

                print(new_rule, file=outfile)
                for ex_id, example in enumerate(examples) :
                    subst = self.head.unify( example )
                    queries.append( ( ext, ex_id, new_head.assign(subst) ) )

            # 3) Write out queries
            for ext, ex_id, atom in queries :
                print('query(%s).' % (atom,), file=outfile)
                         
#            raise NotImplementedError('call ProbLog here...')
            # 4) Call ProbLog
            
            import problog
            
            engine = problog.ProbLogEngine.create([])
            logger = problog.Logger(False)
            dir_out = '/tmp/pf/'
            with problog.WorkEnv(dir_out,logger, persistent=problog.WorkEnv.ALWAYS_KEEP) as env :
                probs = engine.execute(outfile, env) # TODO call problog
            
            print(probs)
            
            # 5) Read out probabilities and return for each literal (lit, [ example scores ]) 
            score_list = []
            prev_ext = None
            for ext, ex_id, atom in queries :
                p = probs.get(atom, False)
                if prev_ext != None and prev_ext != ext :
                    yield prev_ext, score_list
                    score_list = []
                prev_ext = ext
                score_list.append(p)
            yield prev_ext, score_list
        else :
            pass # return nothing

    def evaluate_extensions(self, kb, examples, extensions) :
        prob = []
        nonprob = []
        for ext in extensions :
            if self.body and kb.is_probabilistic(ext) :
                prob.append(ext)
            else :
                nonprob.append(ext)
                if self.body :
                     prob.append(ext)
                    
        from itertools import chain
        # TODO this reorders extensions !!!
        return chain(self.evaluate_extensions_non_prob(kb, examples, nonprob), self.evaluate_extensions_prob(kb, examples, prob))
        
    def extract_variables(self, literals) :
        names = set([])
        result = defaultdict(set)
        for lit in literals :
            lit_vars = lit.variables()
            for t in lit_vars :
                result[t] |= lit_vars[t]
                names |= lit_vars[t]
        return result, names

    def _build_refine_one(self, kb, positive, arg_type, arg_mode, defined_vars) :
        if arg_mode in ['+','-'] :
            for var in self.variables[arg_type] :
                yield var
        if arg_mode == '-' and positive :
            defined_vars.append(True)
            yield kb.newVar()
            defined_vars.pop(-1)
        if arg_mode == 'c' :
            if positive :
                for val in kb.types[arg_type] :
                    yield val
            else :
                yield '_'

    def _build_refine(self, kb, positive, arg_info, defined_vars, use_vars) :
        if arg_info :
            for arg0 in self._build_refine_one(kb, positive, arg_info[0][0], arg_info[0][1], defined_vars) :
                if use_vars != None and arg0 in use_vars :
                    use_vars1 = None
                else :
                    use_vars1 = use_vars 
                for argN in self._build_refine(kb, positive, arg_info[1:], defined_vars, use_vars1) :
                    yield [arg0] + argN
        else :
            if use_vars == None :
                yield []
        
    def refine(self, kb, update=False) :
        if update :
            if not self.body : return
            d, old_vars = self.extract_variables([self.head] + self.body[:-1])
            d, new_vars = self.extract_variables([self.body[-1]])
            use_vars = new_vars - old_vars
            if not use_vars : return
            #use_vars = None
            #print >> sys.stderr, use_vars
        else :
            use_vars = None
        
       # print 'VARS', self.variables        
        for pred_id in kb.modes :
            pred_name = pred_id[0]
            arg_info = list(zip(kb.argtypes(*pred_id), kb.modes[pred_id]))
            for args in self._build_refine(kb, True, arg_info, [], use_vars) :
                yield Literal(kb, pred_name, args)

        for pred_id in kb.modes :
            pred_name = pred_id[0]
            arg_info = list(zip(kb.argtypes(*pred_id), kb.modes[pred_id]))
            for args in self._build_refine(kb, False, arg_info, [], use_vars) :
                yield Literal(kb, '\+' + pred_name, args)
            
    def __str__(self) :
        return str(self.head) + ' :- ' + ', '.join(map(str,self.body)) + '.'
        
    def __add__(self, lit) :
        return Rule(self.head, self.body + [lit])
        
    def __len__(self) :
        return len(self.body)
        
    def countNegated(self) :
        return sum( b.isNegated() for b in self.body )
        
    def __cmp__(self, other) :
        if other == None :
            return 1
        elif len(self) > len(other) :
            return -1
        elif len(self) < len(other) :
            return 1
        elif self.countNegated() < other.countNegated() :
            return 1
        elif self.countNegated() > other.countNegated() :
            return -1
        else :
            return 0



def read_file(filename, idtypes=[]) :

    import re 
    
    line_regex = re.compile( "((?P<type>(base|modes|learn))\()?\s*(?P<name>\w+)\((?P<args>[^\)]+)\)\)?." )
    
    kb = FactDB(idtypes)

    with open(filename) as f :
        for line in f :        
            if line.strip().startswith('%') :
                continue
            m = line_regex.match(line.strip())
            if m : 
                ltype, pred, args = m.group('type'), m.group('name'), list(map(lambda s : s.strip(), m.group('args').split(',')))
                
                if ltype == 'base' :
                    kb.register_predicate(pred, args)                    
                elif ltype == 'modes' :
                    kb.add_mode(pred,args)                    
                elif ltype == 'learn' :
                    kb.add_learn(pred,args)                
                else :
                    kb.add_fact(pred,args)
                                
    return kb

def test(args=[]) :

    kb = read_file(args[0])
    from learn import learn, RuleSet

    print ("# loaded dataset '%s'" % (args[0]))
#    print "# ", kb.modes
    
    stop = False
    while not stop :
        try :
            s = raw_input('?- ')
            l = Literal.parse(kb, s)
            print ('\n'.join(map(str,kb.query(l, {}))))
        except EOFError :
            break
        except KeyboardInterrupt :
            break
        
def main(args=[]) :



    for filename in args :

        kb = read_file(filename)

        from learn import learn, Hypothesis, SETTINGS, Score

        varnames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        targets = kb.learn
    
        filename = os.path.split(filename)[1]
    
        Log.LOG_FILE=open(filename+'.xml', 'w')

        with Log('log') :
            for pred, args in targets :
              with Timer('learning time') as t :
        
                kb.idtypes = args
                kb.reset_examples()
        
                target = Literal(kb, pred, varnames[:len(args)] )
        
                print('==> LEARNING CONCEPT:', target)
                with Log('learn', input=filename, target=target, **vars(SETTINGS)) :
                    H = learn(Hypothesis(Rule, Score, target, kb))   
    
                    print(H)
                    print(H.score, H.score.globalScore)
    
                    with Log('result', time=t.elapsed_time) as l :
                        if l.file :
                            print(H, file=l.file)
    
if __name__ == '__main__' :
    import sys    
    if len(sys.argv) > 1 and sys.argv[1] == 'test' :
        test(sys.argv[2:])
    else :
        main(sys.argv[1:])
    
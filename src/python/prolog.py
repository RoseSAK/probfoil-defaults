#! /usr/bin/env python

from collections import namedtuple, defaultdict
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
                
    def __str__(self) :
        return '%s(%s)' % (self.functor, ','.join(self.args))

    def __repr__(self) :
        return '%s(%s)' % (self.functor, ','.join(self.args))

    def __hash__(self) :
        return hash(str(self))

    def __eq__(self, other) :
        return str(self) == str(other)
        
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
        
    def newVar(self) :
        self.newvar_count += 1
        return 'X_' + str(self.newvar_count)
            
    def register_predicate(self, name, args) :
        arity = len(args)
        identifier = (name, arity)
        
        if not identifier in self.predicates :
            index = [ defaultdict(set) for i in range(0,arity) ] 
            self.predicates[identifier] = (index, [], args, [])
            
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

class Rule(object) :
    
    def __init__(self, head, body=[]) :
        self.head = head
        self.body = body
        
        self.variables, dummy = self.extract_variables([self.head] + self.body)
        self.varcount = sum(map(len,self.variables))
        
    def evaluate(self, kb, example) :
        subst = self.head.unify( example )
        tv_head = true(kb.query([self.head],subst))
        tv_body = true(kb.query(self.body, subst) )
        return tv_head, tv_body
        
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
            arg_info = zip(kb.argtypes(*pred_id), kb.modes[pred_id])
            for args in self._build_refine(kb, True, arg_info, [], use_vars) :
                yield Literal(kb, pred_name, args)

        for pred_id in kb.modes :
            pred_name = pred_id[0]
            arg_info = zip(kb.argtypes(*pred_id), kb.modes[pred_id])
            for args in self._build_refine(kb, False, arg_info, [], use_vars) :
                yield Literal(kb, '\+' + pred_name, args)
            
    def __str__(self) :
        return str(self.head) + ' :- ' + ', '.join(map(str,self.body))
        
    def __add__(self, lit) :
        return Rule(self.head, self.body + [lit])
        
    def __len__(self) :
        return len(self.body)
        
    def countNegated(self) :
        return sum( b.isNegated() for b in self.body )
        

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
                ltype, pred, args = m.group('type'), m.group('name'), map(lambda s : s.strip(), m.group('args').split(','))
                
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

    print "# loaded dataset '%s'" % (args[0])
#    print "# ", kb.modes
    
    stop = False
    while not stop :
        try :
            s = raw_input('?- ')
            l = Literal.parse(kb, s)
            print '\n'.join(map(str,kb.query(l, {})))
        except EOFError :
            break
        except KeyboardInterrupt :
            break
        
def main(args=[]) :

    kb = read_file(args[0])

    from learn import learn, RuleSet, Timer, Log

#    targets = [ 'mammal', 'bird', 'fish', 'reptile', 'amphibian', 'invertebrate']

    varnames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    targets = kb.learn
    
    
    
    Log.LOG_FILE=open(os.path.split(args[0])[1]+'.xml', 'w')
#    Log.LOG_FILE = None

    with Log('log') :
        for pred, args in targets :
          with Timer('learning time') :
        
            kb.idtypes = args
            kb.reset_examples()
        
            target = Literal(kb, pred, varnames[:len(args)] )
        
            print '==> LEARNING CONCEPT:', target 
            with Log('learn', target=target) :
                H = learn(RuleSet(Rule, target, kb))   
    
            print H
            print H.TP, H.TN, H.FP, H.FN
    
    
if __name__ == '__main__' :
    import sys    
    if len(sys.argv) > 1 and sys.argv[1] == 'test' :
        test(sys.argv[2:])
    else :
        main(sys.argv[1:])
    
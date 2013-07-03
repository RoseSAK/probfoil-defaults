#! /usr/bin/env python

from collections import namedtuple, defaultdict
import re

def is_var(arg) :
    return arg == None or arg[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_'

def strip_negation(name) :
    if name.startswith('\+') :
        return name[2:].strip()
    else :
        return name

class Literal(object) :
    
    def __init__(self, kb, functor, args) :
        self.functor = functor
        self.args = args
        self.kb = kb
                
    def __str__(self) :
        return '%s(%s)' % (self.functor, ','.join(self.args))

    def __hash__(self) :
        return hash(str(self))

    def __eq__(self, other) :
        return str(self) == str(other)
        
    def unify(self, ground_args) :
        
        result = {}
        for ARG, arg in zip(self.args, ground_args) :
            if is_var(ARG) and not ARG == '_' :
                result[ARG] = arg
        return result  
        
    def assign(self, subst) :
        new_args = []
        for arg in self.args :
            if arg in subst :
                new_args.append(subst[arg])
            else :
                new_args.append(arg)
        return Literal(self.kb, self.functor, new_args)
        
    def variables(self) :
        result = defaultdict(list)
        types = self.kb.argtypes(strip_negation(self.functor), len(self.args))
        for tp, arg in zip(types, self.args) :
            if is_var(arg) and not arg == '_' :
                result[tp].append(arg)
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
            
    def register_predicate(self, name, args) :
        arity = len(args)
        identifier = (name, arity)
        
        if not identifier in self.predicates :
            index = [ defaultdict(list) for i in range(0,arity) ] 
            values = []
            self.predicates[identifier] = (index, values, args)
            
    def add_mode(self, name, args) :
        arity = len(args)
        identifier = (name, arity)
        self.modes[identifier] = args
        
    def add_learn(self, name, args) :
        self.learn.append( (name, args) )
        
    def add_fact(self, name, args) :
        arity = len(args)
       # self.register_predicate(name, arity)
        identifier = (name, arity)
        index, values, types = self.predicates[identifier]
        
        arg_index = len(values)
        values.append( args )

        for i, arg in enumerate(args) :
            self.types[types[i]].add(arg)
            index[i][arg].append(arg_index)
            
    def argtypes(self, name, arity) :
        identifier = (name, arity)
        return self.predicates[identifier][2]
        
    def find_fact(self, name, args) :
        if name.startswith('\+') :
           # raise NotImplementedError("No support for negative literals yet!")
            negated = True
            name = name[2:].strip()
        else :
            negated = False

        arity = len(args)
        identifier = (name, arity)
        
        index, values = self.predicates[identifier][:2]
        result = set(range(0, len(values))) 
        for i,arg in enumerate(args) :
            if not is_var(arg) :
                arg_index = set(index[i][arg])
                result &= arg_index
                
        if negated :
            if len(result) == 0 :
                return [ args ]
            else :
                return []
        else :
            return [ values[i] for i in result ]  
        
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

    def query(self, literals, substitution) :
        if not literals :  # reached end of query
            yield substitution
        else :
            head, tail = literals[0], literals[1:]
            head_ground = head.assign(substitution)     # assign known variables
            
            for match in self.find_fact(head_ground.functor, head_ground.args) :
                # find match for head
                new_substitution = dict(substitution)
                new_substitution.update( head_ground.unify( match ) )
                for sol in self.query(tail, new_substitution) :
                    yield sol
    
    def test(self, literals, substitution) :
        for sol in self.query(literals, substitution) :
            return True
        return False


class Rule(object) :
    
    def __init__(self, head, body=[]) :
        self.head = head
        self.body = body
        
        self.variables = self.extract_variables()
        self.varcount = sum(map(len,self.variables))
        
    def evaluate(self, kb, example) :
        substitute = self.head.unify( example )
        ground_head = self.head.assign( substitute )
        
        tv_head = kb.test([self.head], self.head.unify( example ))
        
        tv_body = kb.test(self.body, substitute)
        return tv_head, tv_body
        
    def extract_variables(self) :
        result = self.head.variables()
        for lit in self.body :
            result.update( lit.variables() )
        return result

    def build_refine_one(self, kb, positive, arg_type, arg_mode, defined_vars) :
        if arg_mode in ['+','-'] :
            for var in self.variables[arg_type] :
                yield var
        if arg_mode == '-' :
            defined_vars.append(True)
            yield '_' + str(self.varcount + len(defined_vars))
            defined_vars.pop(-1)
        if arg_mode == 'c' :
            if positive :
                for val in kb.types[arg_type] :
                    yield val
            else :
                yield '_'

    def build_refine(self, kb, positive, arg_info, defined_vars) :
        if arg_info :
            for arg0 in self.build_refine_one(kb, positive, arg_info[0][0], arg_info[0][1], defined_vars) :
                for argN in self.build_refine(kb, positive, arg_info[1:], defined_vars) :
                    yield [arg0] + argN
        else :
            yield []
        
        
        
    def refine(self, kb) :
        for pred_id in kb.modes :
            pred_name = pred_id[0]
            arg_info = zip(kb.argtypes(*pred_id), kb.modes[pred_id])
            for args in self.build_refine(kb, True, arg_info, []) :
                yield Literal(kb, pred_name, args)

            for args in self.build_refine(kb, False, arg_info, []) :
                yield Literal(kb, '\+' + pred_name, args)
            
       
    
    def __str__(self) :
        return str(self.head) + ' <- ' + ', '.join(map(str,self.body))
        
    # def __iadd__(self, lit) :
    #     self.body.append(lit)
    #     return self
        
    def __add__(self, lit) :
        return Rule(self.head, self.body + [lit])
        

def read_file(filename, idtypes=[]) :

    import re 
    
    regex_base = re.compile('base\((?P<name>\w+)\((?P<args>[^\)]+)\)\).')
    regex_mode = re.compile('modes\((?P<name>\w+)\((?P<args>[^\)]+)\)\).')
    regex_learn = re.compile('learn\((?P<name>\w+)\((?P<args>[^\)]+)\)\).')
    
    regex = re.compile('(?P<name>\w+)\((?P<args>[^\)]+)\).')

    kb = FactDB(idtypes)

    with open(filename) as f :
        for line in f :
            if line.strip().startswith('%') :
                continue
            
            m = regex_base.match(line.strip())
            if m :
                pred = m.group('name')
                args = map(lambda s : s.strip(), m.group('args').split(','))
                kb.register_predicate(pred, args)
            else :
                m = regex_mode.match(line.strip())
                if m :
                    pred = m.group('name')
                    args = map(lambda s : s.strip(), m.group('args').split(','))
                    kb.add_mode(pred,args)
                else :
                    m = regex_learn.match(line.strip())
                    if m :
                        pred = m.group('name')
                        args = map(lambda s : s.strip(), m.group('args').split(','))
                        kb.add_learn(pred,args)
                    else :
                        m = regex.match(line.strip())
                        if m :
                            pred = m.group('name')
                            args = map(lambda s : s.strip(), m.group('args').split(','))
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

    from learn import learn, RuleSet, Timer

#    targets = [ 'mammal', 'bird', 'fish', 'reptile', 'amphibian', 'invertebrate']

    varnames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    targets = kb.learn

    for pred, args in targets :
      with Timer('learning time') :
        
        kb.idtypes = args
        kb.reset_examples()
        
        target = Literal(kb, pred, varnames[:len(args)] )
        print '==> LEARNING CONCEPT:', target 
        H = learn(RuleSet(Rule, target, kb))   
    
        print H
        print H.TP, H.TN, H.FP, H.FN
    
if __name__ == '__main__' :
    import sys    
    if len(sys.argv) > 1 and sys.argv[1] == 'test' :
        test(sys.argv[2:])
    else :
        main(sys.argv[1:])
    
#! /usr/bin/env python

from collections import namedtuple, defaultdict

class Literal(object) :
    
    def __init__(self, functor, args) :
        self.functor = functor
        self.args = args
                
    def __str__(self) :
        return '%s(%s)' % (self.functor, ','.join(self.args))

    def __hash__(self) :
        return hash(str(self))

    def __eq__(self, other) :
        return str(self) == str(other)
        
    def unify(self, ground_args) :
        
        result = {}
        for ARG, arg in zip(self.args, ground_args) :
            if ARG.startswith('_') :
                result[ARG] = arg
        return result
        
    def assign(self, subst) :
        new_args = []
        for arg in self.args :
            if arg in subst :
                new_args.append(subst[arg])
            else :
                new_args = arg
        return Literal(self.functor, new_args)

class FactDB(object) :
    
    def __init__(self, idpred) :
        self.predicates = {}
        self.idpred = idpred
        self.__examples = None
            
    def register_predicate(self, name, arity) :
        identifier = (name, arity)
        
        if not identifier in self.predicates :
            index = [ defaultdict(list) for i in range(0,arity) ] 
            values = []
            self.predicates[identifier] = (index, values)
        
    def add_fact(self, name, args) :
        arity = len(args)
        self.register_predicate(name, arity)
        identifier = (name, arity)
        index, values = self.predicates[identifier]
        
        arg_index = len(values)
        values.append( args )

        for i, arg in enumerate(args) :
            index[i][arg].append(arg_index)
        
    def find_fact(self, name, args) :
        arity = len(args)
        identifier = (name, arity)
        
        index, values = self.predicates[identifier]
        result = set(range(0, len(values))) 
        for i,arg in enumerate(args) :
            if arg != None :
                arg_index = set(index[i][arg])
                result &= arg_index
        return [ values[i] for i in result ]

    def __str__(self) :
        s = ''
        for name, arity in self.predicates :
            for args in self.predicates[(name,arity)][1] :
                s += '%s(%s)' % (name, ','.join(args)) + '.\n'
        return s
        
    def _get_examples(self) :
        if self.__examples == None :
            name, arity = self.idpred
            self.__examples = self.find_fact(name,[None]*arity )
        return self.__examples
    
    examples = property(_get_examples)
        
    def __getitem__(self, index) :
        return self.examples[index]
        
    def __len__(self) :
        return len(self.examples)
        
    def __iter__(self) :
        return iter(range(0,len(self.examples)))


class Rule(object) :
    
    def __init__(self, head, body=[]) :
        self.head = head
        self.body = body
        
    def evaluate(self, kb, example) :
        substitute = self.head.unify( example )
        ground_head = self.head.assign( substitute )
        
        tv_head = len(kb.find_fact(ground_head.functor, ground_head.args)) > 0
        tv_body = self.evaluate_body(kb, self.body, substitute)
        return tv_head, tv_body
        
    def evaluate_body(self, kb, body, substitution) :
        
        body_first = body[0]
        body_rest = body[1:]
        
        ground_body = body_first.head.assign( substitution )
        
        for match in kb.find_fact(ground_body.functor, ground_body.args) :
            new_substitution = dict(substitution)
            new_substitution.update( self.ground_body.assign( match ) )
            # call recursive
        
    def refine(self, kb) :
        all_fields = set([ f for f in kb.fields() if f[0] != '#' ] )
        my_fields = set([ lit.field for lit in self.body+[self.head] ])
        
        new_fields = all_fields - my_fields
        
        for fld in new_fields :
            for val in kb.values(fld) :
                yield Literal(kb, fld, val)
    
    def __str__(self) :
        return str(self.head) + ' <- ' + ', '.join(map(str,self.body))
        
    def __iadd__(self, lit) :
        self.body.append(lit)
        return self
        
    def __add__(self, lit) :
        return Rule(self.head, self.body + [lit])
        

def read_file(filename, idpred) :

    import re 
    
    regex = re.compile('(?P<name>\w+)\((?P<args>[^\)]+)\).')

    kb = FactDB(idpred)

    with open(filename) as f :
        for line in f :
            m = regex.match(line.strip())
            if m :
                pred = m.group('name')
                args = map(lambda s : s.strip(), m.group('args').split(','))
                kb.add_fact(pred,args)
    return kb


    
def test(args=[]) :

    kb = read_file(args[0], ('animal',1))
    
    from learn import learn, RuleSet

    target = Literal('mammal', ['_X'])
    print '==> LEARNING CONCEPT:', target 
    print learn(RuleSet(Rule, target, kb))   
#        break 
    
    
#    print data.find_fact('has_legs', [None, '4'])
    
if __name__ == '__main__' :
    import sys    
    if len(sys.argv) > 1 and sys.argv[1] == 'test' :
        test(sys.argv[2:])
    else :
        main(sys.argv[1:])
    
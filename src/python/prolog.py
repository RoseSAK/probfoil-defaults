#! /usr/bin/env python

from collections import namedtuple, defaultdict
import re

def is_var(arg) :
    return arg == None or arg[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_'


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
        return Literal(self.functor, new_args)
        
    @classmethod
    def parse(cls, string) :
        regex = re.compile('\s*(?P<name>[^\(]+)\((?P<args>[^\)]+)\)[,.]?\s*')
        result = []
        for name, args in regex.findall(string) :
            result.append( Literal( name , map(lambda s : s.strip(), args.split(','))) )
        return result
        
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
        if name.startswith('\+') :
           # raise NotImplementedError("No support for negative literals yet!")
            negated = True
            name = name[2:].strip()
        else :
            negated = False

        arity = len(args)
        identifier = (name, arity)
        
        index, values = self.predicates[identifier]
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
        
    def evaluate(self, kb, example) :
        substitute = self.head.unify( example )
        ground_head = self.head.assign( substitute )
        
        tv_head = kb.test([self.head], self.head.unify( example ))
        
        tv_body = kb.test(self.body, substitute)
        return tv_head, tv_body
        
    # def evaluate_body(self, kb, body, substitution) :
    #     if not body : 
    #         return True
    #     
    #     
    #     body_first = body[0]
    #     body_rest = body[1:]
    #     
    #     ground_body = body_first.assign( substitution )
    #     
    #     for match in kb.find_fact(ground_body.functor, ground_body.args) :
    #         new_substitution = dict(substitution)
    #         new_substitution.update( ground_body.unify( match ) )
    #         if self.evaluate_body(kb, body_rest, new_substitution) :
    #             return True
    #     return False
        
    def refine(self, kb) :
        yield Literal('\+animal',['X'])
        yield Literal('\+has_hair',['X'])
        yield Literal('\+has_feathers',['X'])
        yield Literal('\+lays_eggs',['X'])
        yield Literal('\+gives_milk',['X'])
        yield Literal('\+is_airborne',['X'])
        yield Literal('\+is_aquatic',['X'])
        yield Literal('\+is_predator',['X'])
        yield Literal('\+is_toothed',['X'])
        yield Literal('\+has_backbone',['X'])
        yield Literal('\+breathes',['X'])
        yield Literal('\+is_venomous',['X'])
        yield Literal('\+has_fins',['X'])
        yield Literal('\+has_legs',['X','_'])
        yield Literal('\+has_tail',['X'])
        yield Literal('\+is_domestic',['X'])
        yield Literal('\+is_catsize',['X'])
        yield Literal('animal',['X'])
        yield Literal('has_hair',['X'])
        yield Literal('has_feathers',['X'])
        yield Literal('lays_eggs',['X'])
        yield Literal('gives_milk',['X'])
        yield Literal('is_airborne',['X'])
        yield Literal('is_aquatic',['X'])
        yield Literal('is_predator',['X'])
        yield Literal('is_toothed',['X'])
        yield Literal('has_backbone',['X'])
        yield Literal('breathes',['X'])
        yield Literal('is_venomous',['X'])
        yield Literal('has_fins',['X'])
        yield Literal('has_legs',['X','2'])
        yield Literal('has_legs',['X','4'])
        yield Literal('has_legs',['X','5'])
        yield Literal('has_legs',['X','6'])    
        yield Literal('has_legs',['X','8'])
        yield Literal('has_tail',['X'])
        yield Literal('is_domestic',['X'])
        yield Literal('is_catsize',['X'])
    
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

    print "# loaded dataset '%s'" % (args[0])
    
    stop = False
    while not stop :
        try :
            s = raw_input('?- ')
            l = Literal.parse(s)
            print '\n'.join(map(str,kb.query(l, {})))
        except EOFError :
            break
        except KeyboardInterrupt :
            break

#    print kb.find_fact(l1.functor, l1.args)

    
def main(args=[]) :

    kb = read_file(args[0], ('animal',1))
    
    from learn import learn, RuleSet

    targets = [ 'mammal', 'bird', 'fish', 'reptile', 'amphibian', 'invertebrate']
#    targets = [ 'fish' ]

    for t in targets :
        target = Literal(t, ['X'])
        print '==> LEARNING CONCEPT:', target 
        H = learn(RuleSet(Rule, target, kb))   
    
        print H
        print H.TP, H.TN, H.FP, H.FN
#        break 
    
    
#    print data.find_fact('has_legs', [None, '4'])
    
if __name__ == '__main__' :
    import sys    
    if len(sys.argv) > 1 and sys.argv[1] == 'test' :
        test(sys.argv[2:])
    else :
        main(sys.argv[1:])
    
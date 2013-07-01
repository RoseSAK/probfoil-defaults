#! /usr/bin/env python

from collections import namedtuple, defaultdict

class Literal(object) :
    
    def __init__(self, kb, field, value) :
        self.kb = kb
        self.field = field
        self.value = value
        
    def __iter__(self) :
        return iter( [ self.field, self.value ] )

    def __str__(self) :
        return self.kb.format( self )

    def __hash__(self) :
        return hash(str(self))

    def __eq__(self, other) :
        return str(self) == str(other)


class SimpleTable(object) :
    
    def __init__(self, header, data) :
        self.__data = data
        self.__header = header
        self.__names = dict( (x,i) for i,x in enumerate(header) )
        self.__values = defaultdict(set)
        
        for index in self :
            example = self[index]
            for field in example :
                self.__values[field].add(example[field])
        
    def __len__(self) :
        return len(self.__data)
        
    def __iter__(self) :
        return iter(range(0,len(self)))

    def __getitem__(self, index) :
        example = self.__data[index]
        return dict( (n, example[i]) for i,n in enumerate(self.__header ))
        
    def fields(self) :
        return self.__header
        
    def values(self, field) :
        return self.__values[field]
        
    def literals(self, index) :
        example = self.__data[index]
        return set( Literal(self, n, example[i]) for i,n in enumerate(self.__header ) )
        
        
    def format(self, literal) :
        field, value = literal
        values = self.values(field)
        if len(values) == 2 and '0' in values and '1' in values :
            if value == '0' :
                return '!' + field
            else :
                return field
        else :
            return field + '(' + value + ')'
    

class Rule(object) :
    
    def __init__(self, head, body=[]) :
        self.head = head
        self.body = body
        
    def evaluate(self, example) :
        p, v = self.head
        tv_head = (example[p] == v) 
        
        for p, v in self.body :
            if example[p] != v :
               return tv_head, False

        return tv_head, True 
        
    def refine(self, kb) :
        all_fields = set([ f for f in kb.fields() if f[0] != '#' ] )
        my_fields = set([ f for f,v in self.body+[self.head] ])
        
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
        

def read_file(filename) :
    line_parse = lambda line : line.strip().split(',')
    with open(filename) as f :
        data = map(line_parse, f.readlines())
        
    return data
    
def test(args=[]) :

    data = read_file(args[0])
    kb = SimpleTable(data[0], data[1:])

    from learn import learn, RuleSet

    field = kb.fields()[-1]
    for value in kb.values(field) :
        target = Literal(kb, field, value)
        print learn(RuleSet(Rule, target, kb))    
    
if __name__ == '__main__' :
    import sys    
    if len(sys.argv) > 1 and sys.argv[1] == 'test' :
        test(sys.argv[2:])
    else :
        main(sys.argv[1:])
    
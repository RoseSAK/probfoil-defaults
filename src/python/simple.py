#! /usr/bin/env python

from collections import namedtuple

Example = list

Literal = namedtuple('Literal',  ['pos', 'val'] )

class KnowledgeBase(list) :
    
    def __init__(self, header, data) :
        super(KnowledgeBase,self).__init__(data)
        
        self.fields = dict( [ (x,i) for i,x in enumerate(header) ] )
        self.values = [ set([ row[col] for row in data ]) for col in range(0,len(data[0])) ]
    
    def field(self, name) :
        return self.fields[name]
    
    def field_values(self, name) :
        return self.values[self.field(name)]
        
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
        all_fields = set([ kb.field(x) for x in kb.fields if kb.field(x) != self.head[0] and kb.field(x) != 0  ])
        my_fields = set([ i for i,v in self.body ])
        
        new_fields = all_fields - my_fields
        
        for fld in new_fields :
            for val in kb.values[fld] :
                yield (fld, val)
    
    def __str__(self) :
        return str(self.head) + ' <- ' + ', '.join(map(str,self.body))
        
    def __iadd__(self, lit) :
        self.body.append(lit)
        return self
        
    def __add__(self, lit) :
        return Rule(self.head, self.body + [lit])
    
class RuleSet(object) :
    
    def __init__(self, target) :
        self.rules = []
        self.target = target
        
    def evaluate_one(self, example) :
        for rule in self.rules :
            h, b = rule.evaluate(example)
            if b : return h, True
            
        h, b = Rule(self.target).evaluate(example)
    
        return h, False
        
    def evaluate(self, data) :
        TP = []
        TN = []
        FP = []
        FN = []
    
        for i, ex in enumerate(data) :
            h, b = self.evaluate_one(ex)
            
            if h :  # positive 
                if b :
                    TP += [i]
                else :
                    FN += [i]
            else :
                if b :
                    FP += [i]
                else : 
                    TN += [i]
        return TP, TN, FP, FN        
        
    def __iadd__(self, rule) :
        self.rules.append(rule)
        return self
        
    def __add__(self, rule) :
        if type(rule) == list :
            raise RuntimeError
        result = RuleSet(self.target)
        result.rules = self.rules + [rule]
        return result
        
    def __str__(self) :
        return '\n'.join(map(str,self.rules))
    
def read_file(filename) :
    line_parse = lambda line : line.strip().split(',')
    with open(filename) as f :
        data = map(line_parse, f.readlines())
        
    return data
    
def test(args=[]) :

    data = read_file('../../data/test.dat')
    kb = KnowledgeBase(data[0], data[1:])

    from learn import learn, SimpleLearn
    # r0 = learn(SimpleLearn(kb,(kb.field('C'),'0')))
    # print 'RESULT:'
    # print r0
    # 
    # r1 = learn(SimpleLearn(kb,(kb.field('C'),'1')))
    # 
    # print 'RESULT:'
    # print r1


    data = read_file('../../data/zoo.dat')
    kb = KnowledgeBase(data[0], data[1:])

    from learn import learn, SimpleLearn

    print learn(SimpleLearn(kb,(kb.field('type'),'1')))    
    print learn(SimpleLearn(kb,(kb.field('type'),'2')))    
    print learn(SimpleLearn(kb,(kb.field('type'),'3')))    
    print learn(SimpleLearn(kb,(kb.field('type'),'4')))    
    print learn(SimpleLearn(kb,(kb.field('type'),'5')))    
    print learn(SimpleLearn(kb,(kb.field('type'),'6')))    
    print learn(SimpleLearn(kb,(kb.field('type'),'7')))        
    
if __name__ == '__main__' :
    import sys    
    if len(sys.argv) > 1 and sys.argv[1] == 'test' :
        test(sys.argv[2:])
    else :
        main(sys.argv[1:])
#! /usr/bin/env python

from collections import namedtuple

Example = list

Literal = namedtuple('Literal',  ['pos', 'val'] )

class KnowledgeBase(object) :
    
    def __init__(self, header, data) :
        super(KnowledgeBase,self).__init__()
        
        self.data = data
        
        self.fields = dict( [ (x,i) for i,x in enumerate(header) ] )
        self.header = header
        self.values = [ set([ row[col] for row in data ]) for col in range(0,len(data[0])) ]
    
    def __getitem__(self, index) :
        return self.data[index]
        
    def __iter__(self) :
        return iter(range(0,len(self.data)))
    
    def field(self, name) :
        return self.fields[name]
    
    def field_values(self, name) :
        return self.values[self.field(name)]


class RuleSet(object) :
    
    def __init__(self, target, data) :
        self.target = target
        self.rules = []
        self.data = data
        
        evaluated = ( [], [] )
        for ex in self.data :
            h, b = Rule(target).evaluate(self.data[ex])
            evaluated[h].append( ex )
        negatives, positives = evaluated
        
        self.P = float(len(positives))
        self.N = float(len(negatives))
        
        self.POS = [positives]  # should be set of positive examples
        self.NEG = [negatives]  # should be set of negative examples
                
    def getTP(self) :
        return float(self.P - self.FN)
    
    def getTN(self) :
        return float(len(self.NEG[0]))
    
    def getFP(self) :
        return float(self.N - self.TN)
    
    def getFN(self) :
        return float(len(self.POS[0]))

    def getTP1(self) :
        return float(self.P - self.FN1)
    
    def getTN1(self) :
        return float(len(self.NEG[0]) + len(self.NEG[-1]))
    
    def getFP1(self) :
        return float(self.N - self.TN1)
    
    def getFN1(self) :
        return float(len(self.POS[0]) + len(self.POS[-1]))

        
    TP = property(getTP)    
    TN = property(getTN)
    FP = property(getFP)
    FN = property(getFN)

    TP1 = property(getTP1)    
    TN1 = property(getTN1)
    FP1 = property(getFP1)
    FN1 = property(getFN1)

        
    def pushRule(self, rule=None) :
        if rule == None : rule = self.newRule()
        
        evaluated = [[],[]]
        for example in self.POS[0] :
            h, b = rule.evaluate(self.data[example])
            evaluated[b].append(example)
        self.POS[0] = evaluated[0]
        self.POS.append( evaluated[1] )
        
        evaluated = [[],[]]
        for example in self.NEG[0] :
            h, b = rule.evaluate(self.data[example])
            evaluated[b].append(example)
        self.NEG[0] = evaluated[0]
        self.NEG.append( evaluated[1] )                    
        
        self.rules.append(rule)

    def popRule(self) :
        self.POS[0] = self.POS[0]+self.POS[-1]
        self.POS.pop(-1)
        
        self.NEG[0] = self.NEG[0]+self.NEG[-1]
        self.NEG.pop(-1)
        
        self.rules.pop(-1)
        
    def pushLiteral(self, literal) :
        # quick implementation
        rule = self.rules[-1]
        self.popRule()
        self.pushRule( rule + literal )
    
    def popLiteral(self) :
        # quick implementation
        rule = self.rules[-1]
        self.popRule()
        rule.body = rule.body[:-1]
        self.pushRule( rule )
        
    def newRule(self) :
        return Rule(self.target)
        
    def __str__(self) :
        return '\n'.join(map(str,self.rules))
        
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
    
# class RuleSet(object) :
#     
#     def __init__(self, target) :
#         self.rules = []
#         self.target = target
#         self.newRule = Rule
#         
#     def evaluate_one(self, example) :
#         for rule in self.rules :
#             h, b = rule.evaluate(example)
#             if b : return h, True
#             
#         h, b = Rule(self.target).evaluate(example)
#     
#         return h, False
#         
#     def evaluate(self, data) :
#         TP = []
#         TN = []
#         FP = []
#         FN = []
#     
#         for i, ex in enumerate(data) :
#             h, b = self.evaluate_one(ex)
#             
#             if h :  # positive 
#                 if b :
#                     TP += [i]
#                 else :
#                     FN += [i]
#             else :
#                 if b :
#                     FP += [i]
#                 else : 
#                     TN += [i]
#         return TP, TN, FP, FN        
#         
#     def __iadd__(self, rule) :
#         self.rules.append(rule)
#         return self
#         
#     def __add__(self, rule) :
#         if type(rule) == list :
#             raise RuntimeError
#         result = RuleSet(self.target)
#         result.rules = self.rules + [rule]
#         return result
#         
#     def __str__(self) :
#         return '\n'.join(map(str,self.rules))
    
def read_file(filename) :
    line_parse = lambda line : line.strip().split(',')
    with open(filename) as f :
        data = map(line_parse, f.readlines())
        
    return data
    
def test(args=[]) :

    data = read_file('../../data/test.dat')
    kb = KnowledgeBase(data[0], data[1:])

    

    data = read_file('../../data/zoo.dat')
    kb = KnowledgeBase(data[0], data[1:])

    from learn import learn

    print learn(RuleSet((kb.field('type'),'1'), kb))    
    print learn(RuleSet((kb.field('type'),'2'), kb))
    print learn(RuleSet((kb.field('type'),'3'), kb))   
    print learn(RuleSet((kb.field('type'),'4'), kb))   
    print learn(RuleSet((kb.field('type'),'5'), kb))   
    print learn(RuleSet((kb.field('type'),'6'), kb))   
    print learn(RuleSet((kb.field('type'),'7'), kb))   
    
    
if __name__ == '__main__' :
    import sys    
    if len(sys.argv) > 1 and sys.argv[1] == 'test' :
        test(sys.argv[2:])
    else :
        main(sys.argv[1:])
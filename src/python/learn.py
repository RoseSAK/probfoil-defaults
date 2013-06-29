#! /usr/bin/env python3


def learn(target, l) :
        
    H = l.RuleSet()
    newH = None
    
    h = target
    while (newH == None or l.globalScore(H) > l.globalScore(H,c)) :
        r = l.Rule(h,[])
        while not l.localStop(H,r) :
            l = argmax(r.refine(),lambda l : l.localScore(H, r+l)) 
            r += l
        n = len(r.body)
        i = argmax( range(1,n), lambda i : l.localScore(H, r.body[:i]) )
        
        c = l.Rule( r.head, r.body[:i] )
        
        newH = H + c
        if (l.globalScore(H) < l.globalScore(newH)) :
            H = newH
    return H
        
def argmax( generator, score ) :
    max_score = None
    max_x = None
    for x in generator :
        current_score = score(x)
        if max_score == None or max_score < current_score :
            max_score = current_score
            max_x = x
    return max_x
    
class SimpleDB(object) :
    
    def __init__(self, examples) :
        self.examples = examples
        
    def __len__(self) :
        return len(self.examples)

class SimpleExample(object):
        
    def __init__(self, values) :
        self.values = values
        
    def evaluate(self, rule) :
        h_pos, h_val = rule.head
        tv_head = self.values[h_pos] == h_val
        for b_pos, b_val in rule.body :
            tv_body = tv_body and self.values[b_pos] == b_val
            if not tv_body : break
        return tv_head, tv_body
    
    
class SimpleLearn(object) :
    
    def __init__(self) :
        self.RuleSet = SimpleRuleSet
        self.Rule = SimpleRule
        self.P = 0
        self.N = 0

    def localStop(self, rs, rule) :
        rs2 = rs + rule
        return ( rs2.TP - rs.TP == 0 ) or ( rs2.FP - rs.FP == 0 )
    
    def localScore(self, ruleset, rule) :
        return self.m_estimate(ruleset, rule) - self.m_estimate(ruleset)
        
    def globalScore(self, rs) :
        return (rs.TP + rs.TN ) / (rs.TP + rs.TN + rs.FP + rs.FN)
        
    def m_estimate(self, rs, m=0) :
        return (rs.TP + m * (self.P / (self.N + self.P))) / (rs.TP + rs.FP + m)
        
class SimpleRuleSet(object) :
    
    def __init__(self) :
        self.rules = []
        self.TP = 0
        self.FP = 0
        self.TN = 0 
        self.FN = 0
        
    def __iadd__(self, rule) :
        self.rules.append(rule)

    def __add__(self, rule) :
        result = SimpleRuleSet()
        result.rules = self.rules + [rule]
        return result 
        
    def __str__(self) :
        return '\n'.join(map(str,rules))
        
        
        
class SimpleRule(object) :
    
    def __init__(self, head, body) :
        self.head = head
        self.body = body
        
    def __add__(self, literal) :
        return SimpleRule(self.head, self.body + literal)
    
    def __iadd__(self, literal) :
        self.body += literal
        
    def __str__(self) :
        return str(head) + ' <- ' + ', '.join(map(str,body))
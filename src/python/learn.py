#! /usr/bin/env python3


def learn(l) :
        
    H = l.RuleSet(l.target)
    newH = None
    
    h = l.target
    stop = False
    while (newH == None or not stop) :
        stop = True
        r = l.Rule(h,[])
        while not l.localStop(H,r) :
            lit = argmax(r.refine(l.data),lambda lt : l.localScore(H, r+lt)) 
            #if lit == None : return H
            r += lit
        n = len(r.body)
        
        i = argmax( range(1,n+1), lambda i : l.localScore(H, l.Rule(l.target,r.body[:i]) ))
        
        c = l.Rule( r.head, r.body[:i] )
        
        newH = H + c
        
        score_H = l.globalScore(H)
        score_nH = l.globalScore(newH)
        
        if (l.globalScore(H) < l.globalScore(newH)) :
            H = newH
            stop = False
    return H
        
def argmax( generator, score ) :
    max_score = None
    max_x = None
    
    refs = list(generator)
    
#    print 'ARGMAX'
    for x in refs :
        current_score = score(x)
#        print 'REF', x, current_score
        if max_score == None or max_score < current_score :
            max_score = current_score
            max_x = x
#    print '==>', max_x, max_score
    return max_x
        
import simple
        
class SimpleLearn(object) :
    
    def __init__(self, data, target) :
        self.RuleSet = simple.RuleSet
        self.Rule = simple.Rule
        self.data = data
        self.target = target
        
        rules = self.RuleSet(target)
        TP, self.N, FN, self.P = map(len,rules.evaluate(data))

    def localStop(self, rs, rule) :
        rs2 = rs + rule
        
        rsTP, rsTN, rsFP, rsFN = map(len,rs.evaluate(self.data))
        rs2TP, rs2TN, rs2FP, rs2FN = map(len,rs2.evaluate(self.data))
        
        # print rs
        # print '---'
        # print rs2
        # 
        # print rsTP, rsTN, rsFP, rsFN,         rs2TP, rs2TN, rs2FP, rs2FN
        
        return ( rs2TP - rsTP == 0 ) or ( rs2FP - rsFP == 0 )
    
    def localScore(self, ruleset, rule) :
        rs2score = self.m_estimate(ruleset+rule)
        rsscore = self.m_estimate(ruleset)
        #print rs2score, rsscore, rule
        return rs2score - rsscore
        
    def globalScore(self, rs) :
        rsTP, rsTN, rsFP, rsFN = map(len,rs.evaluate(self.data))
        return float(rsTP + rsTN ) / (rsTP + rsTN + rsFP + rsFN)
        
    def m_estimate(self, rs, m=1) :
        rsTP, rsTN, rsFP, rsFN = map(len,rs.evaluate(self.data))
        #print rsTP, rsTN, rsFP, rsFN
        return float(rsTP + m * (self.P / (self.N + self.P))) / (rsTP + rsFP + m)
        

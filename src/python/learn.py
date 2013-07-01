#! /usr/bin/env python3

def localStop(H) :
    
    TP_p = H.TP1
    TP_c = H.TP
    
    FP_p = H.FP1
    FP_c = H.FP
    
    return ( TP_c - TP_p == 0 ) or ( FP_c - FP_p == 0 )

def localScore(rs) :
    m = 1
    
    m_estimate_c = (rs.TP + m * (rs.P / (rs.N + rs.P))) / (rs.TP + rs.FP + m) 
    
    m_estimate_p = (rs.TP1 + m * (rs.P / (rs.N + rs.P))) / (rs.TP1 + rs.FP1 + m) 
    
    return m_estimate_c - m_estimate_p
    
def globalScore(rs) :
    return (rs.TP + rs.TN ) / (rs.TP + rs.TN + rs.FP + rs.FN)
    
def learn(H) :

    score_H = globalScore(H)

    while True :
        H.pushRule()
        while not localStop(H) :
            lit = best_literal( H, H.refine(), lambda H : localScore(H) ) 
            #if lit == None : return H
            H.pushLiteral(lit)

        
        # FIXME pruning step
        # n = len(r.body)
        # 
        # i = argmax( range(1,n+1), lambda i : localScore(H, H.newRule(target,r.body[:i]), data ))
        # 
        # c = H.newRule( r.head, r.body[:i] )        
        # newH = H + c
        
        score_nH = globalScore(H)
        
        print score_H, score_nH, H.rules[-1]
        if (score_H >= score_nH) :
            H.popRule()
            break
        else :
            score_H = score_nH
    return H

def best_literal( H, generator, score_func ) :
    max_score = None
    max_x = None
    
    for lit in generator :
        H.pushLiteral(lit)
        current_score = score_func(H)
        if max_score == None or max_score < current_score :
            max_score, max_x = current_score, lit
        H.popLiteral()
#    lit = argmax(r.refine(data), lambda lt : localScore(H, r+lt, data)) 
    return max_x


class RuleSet(object) :
    
    def __init__(self, RuleType, target, data) :
        self.target = target
        self.rules = []
        self.data = data
        self.Rule = RuleType
        
        evaluated = ( [], [] )
        for ex in self.data :
            h, b = self.Rule(target).evaluate(self.data[ex])
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

    def refine(self) :
        # Generate literals that are in FP of last rule
        TP_examples = self.POS[-1]
        FN_examples = self.POS[0]
        literals = set([])
        for x in TP_examples + FN_examples :
#            print self.data.literals(x)
            literals |= self.data.literals(x)
            
        result = set(self.rules[-1].refine(self.data))
        result = literals & result
        print len(result), 'refinements', self.TP, self.FN
        return result
        
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
        return self.Rule(self.target)
        
    def __str__(self) :
        return '\n'.join(map(str,self.rules))
        
    

        
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
        


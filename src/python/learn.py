#! /usr/bin/env python3

from collections import namedtuple
import time, sys

BEAM_SIZE = 5

def localStop(H) :
    
    TP_p = H.TP1
    TP_c = H.TP
    
    FP_p = H.FP1
    FP_c = H.FP
    
    return ( TP_c - TP_p == 0 ) or ( FP_c - FP_p == 0 )

def localScore(rs) :
    m = 10
    
    m_estimate_c = (rs.TP + m * (rs.P / (rs.N + rs.P))) / (rs.TP + rs.FP + m) 
    
    m_estimate_p = (rs.TP1 + m * (rs.P / (rs.N + rs.P))) / (rs.TP1 + rs.FP1 + m) 
    
    return m_estimate_c - m_estimate_p
    
def globalScore(rs) :
    return (rs.TP + rs.TN ) / (rs.TP + rs.TN + rs.FP + rs.FN)
    
def learn(H) :
    score_H = globalScore(H)
    while True :
        rule = best_clause( H )
        H.pushRule(rule)        
        score_nH = globalScore(H)
        
        if (score_H >= score_nH) :
            H.popRule()
            break
        else :
            score_H = score_nH
    return H

def best_clause( H, beam_size = BEAM_SIZE ) :
    beam = Beam(beam_size)
    beam.push((False,H.newRule()) , None)
    
    while True :
        print >> sys.stderr, beam
        
        next_beam = Beam(beam_size)
        
        keep = []
        for s, h_rule in beam :
            handled, rule = h_rule
            if handled or (s != None and s[2] == 0) : # no need to extend rules if FP is already 0
               keep.append((s, rule)) 
            else :
                H.pushRule(rule)
                sub_beam = best_literal( H, H.refine(), lambda H : localScore(H), 5 )
                H.popRule()
                better_child = False
                for score, lit in sub_beam :
                    if s == None or score > s :
                        better_child = True
                        next_beam.push( (False, rule + lit), score + (-len(rule.body),) )
            
                if not better_child :
                    keep.append((s, rule))

        if not next_beam.content :
            break
        else :
            beam = next_beam
            for s,r in keep :
                beam.push((True,r),s)
    
    return beam.content[0][1][1]

class Beam(object) :
    
    def __init__(self, size) :
        self.size = size
        self.content = []
       
    def __iter__(self) :
        return iter(self.content)
        
    def push(self, obj, score) :
        if len(self.content) == self.size and score < self.content[-1][0] : return
        
        p = len(self.content) - 1
        self.content.append( (score, obj) )
        while p >= 0 and self.content[p][0] < self.content[p+1][0] :
            self.content[p], self.content[p+1] = self.content[p+1], self.content[p] # swap elements
            p = p - 1
        
        while len(self.content) > self.size :
            self.content.pop(-1)
    
    def pop(self) :
        self.content = self.content[1:]
        
    def __str__(self) :
        r = 'BEAM <<<\n'
        for s, c in self.content :
            r += str(c[1]) + ': ' + str(s) + '\n'
        r += '>>>'
        return r
        
def best_literal( H, generator, score_func , beam_size) :
    beam = Beam(beam_size)
    
    EvalStats = namedtuple('EvalStats', ['TP', 'TN', 'FP', 'FN', 'P', 'N', 'TP1', 'TN1', 'FP1', 'FN1' ]  )    
    for lit in generator :
        
        posM, negM = H.testLiteral(lit)
        stats = EvalStats(H.TP - posM, H.TN + negM, H.FP - negM, H.FN + posM, H.P, H.N, H.TP1, H.TN1, H.FP1, H.FN1 )

        if negM > 0 and H.TP - posM > H.TP1 :
            current_score = score_func(stats), stats.TP, stats.FP
            print >> sys.stderr, 'accepted', current_score, H.rules[-1], lit, negM, posM

            beam.push(lit, current_score)
        else :
            print >> sys.stderr, 'rejected', H.rules[-1], lit, negM, posM
    return beam

class Timer(object) :
    
    def __init__(self, desc) :
        self.desc = desc
    
    def __enter__(self) :
        self.start = time.time()
        
    def __exit__(self, *args) :
        t = time.time()
        print ( '%s: %.5fs' % (self.desc, t-self.start ))

class RuleSet(object) :
    
    def __init__(self, RuleType, target, data) :
        self.target = target
        self.rules = []
        self.data = data
        self.Rule = RuleType
        
        evaluated = ( [], [] )
        for ex in self.data :
            h, b = self.Rule(target).evaluate(self.data, self.data[ex])
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
        return self.rules[-1].refine(self.data)
        
    def pushRule(self, rule=None) :
        if rule == None : rule = self.newRule()
        
        evaluated = [[],[]]
        for example in self.POS[0] :
            h, b = rule.evaluate(self.data, self.data[example])
            evaluated[b].append(example)
        self.POS[0] = evaluated[0]
        self.POS.append( evaluated[1] )
        
        evaluated = [[],[]]
        for example in self.NEG[0] :
            h, b = rule.evaluate(self.data, self.data[example])
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
        
    def testLiteral(self, literal) :
        current_rule = self.rules[-1]
        
        new_rule = current_rule + literal
        # can only move examples from self.XXX[-1] to self.XXX[0]
        
        posMoved = 0
        for ex in self.POS[-1] :
            h, b = new_rule.evaluate(self.data, self.data[ex])
            if not b :
                posMoved += 1

        negMoved = 0
        for ex in self.NEG[-1] :
            h, b = new_rule.evaluate(self.data, self.data[ex])
            if not b :
                negMoved += 1
        
        return posMoved, negMoved
        
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

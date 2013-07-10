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
    #beam.push((False,H.newRule()) , None)
    
    rule = H.newRule()
    refinements = [ (0,r) for r in rule.refine(H.data) ]
    

    beam.push( rule, refinements , None )

    while beam.has_active() :
        new_beam = Beam(beam_size)
        for score, rule, refs in beam :
            new_beam.push( rule, None, score )
            H.pushRule(rule)
            new_refs = update_refinements(H, refs, localScore)
            print >> sys.stderr, '>>', new_refs
            for i, score_ref in enumerate(new_refs) :
                new_score, ref = score_ref
                if new_score <= score or not new_beam.push( rule + ref, new_refs, new_score ) : # was new_refs[i+1:]
                    break  # current ref does not have high enough score -> next ones will also fail
            print >> sys.stderr, new_beam
            H.popRule()
        beam = new_beam
        
    print >> sys.stderr, 'FOUND RULE',beam.content[0][1]
    return beam.content[0][1]

class Beam(object) :
    
    def __init__(self, size) :
        self.size = size
        self.content = []
       
    def __iter__(self) :
        return iter(self.content)
        
    def push(self, obj, active, score) :
        if len(self.content) == self.size and score < self.content[-1][0] : return False
        
        is_last = True
        
        p = len(self.content) - 1
        self.content.append( (score, obj, active) )
        while p >= 0 and self.content[p][0] < self.content[p+1][0] :
            self.content[p], self.content[p+1] = self.content[p+1], self.content[p] # swap elements
            p = p - 1
            is_last = False
        
        popped_last = False
        while len(self.content) > self.size :
            self.content.pop(-1)
            popped_last = True
            
        #print is_last, popped_last, self
            
        return not (is_last and popped_last)
    
    def peak_active(self) :
        i = 0
        while i < len(self.content) :
            if self.content[i][2] :
                yield self.content[i]
                i = 0
            else :
                i += 1
                
    def has_active(self) :
        for s, r, act in self :
            if act : return True
        return False
    
    def pop(self) :
        self.content = self.content[1:]
        
    def __str__(self) :
        res = 'BEAM <<<\n'
        for s, c, r in self.content :
            res += str(c) + ': ' + str(s) + '\n' # ' | ' + str(r) + '\n'
        res += '>>>'
        return res
        
def update_refinements(H, refine, score_func) :
    if refine == None :
        return []
    
    literals = []
    
    new_refine = [ (0,r) for r in H.refine(update=False) ]
    
#    if new_refine : print >> sys.stderr, 'NEW REFINEMENTS', new_refine
    
 #   refine += new_refine
#    
#    refine = set(refine)
    refine = new_refine
    
    for s, lit in refine :
        
 #       posM, negM = 
        stats = H.testLiteral(lit)
        
        negM = stats.TN - H.TN
        posM = H.TP - stats.TP
        
#        stats = EvalStats(H.TP - posM, H.TN + negM, H.FP - negM, H.FN + posM, H.P, H.N, H.TP1, H.TN1, H.FP1, H.FN1 )

        if negM > 0 and H.TP - posM > H.TP1 :
            current_score = score_func(stats), stats.TP, stats.FP
            print >> sys.stderr, 'accepted', H.rules[-1], lit, negM, posM, stats, stats[-1], current_score
            literals.append( (current_score[0], lit) )
        else :
            # literal doesn't cover true positives or it doesn't eliminate false positives
            print >> sys.stderr, 'rejected', H.rules[-1], lit, negM, posM, stats, stats[-1]
    return list(reversed(sorted(literals)))        
        
        
# def best_literal( H, generator, score_func , beam_size) :
#     beam = Beam(beam_size)
#     
#     EvalStats = namedtuple('EvalStats', ['TP', 'TN', 'FP', 'FN', 'P', 'N', 'TP1', 'TN1', 'FP1', 'FN1' ]  )    
#     for lit in generator :
#         
#         posM, negM = H.testLiteral(lit)
#         stats = EvalStats(H.TP - posM, H.TN + negM, H.FP - negM, H.FN + posM, H.P, H.N, H.TP1, H.TN1, H.FP1, H.FN1 )
# 
#         if negM > 0 and H.TP - posM > H.TP1 :
#             current_score = score_func(stats), stats.TP, stats.FP
#             print >> sys.stderr, 'accepted', current_score, H.rules[-1], lit, negM, posM
# 
#             beam.push(lit, current_score)
#         else :
#             print >> sys.stderr, 'rejected', H.rules[-1], lit, negM, posM
#     return beam

class Timer(object) :
    
    def __init__(self, desc) :
        self.desc = desc
    
    def __enter__(self) :
        self.start = time.time()
        
    def __exit__(self, *args) :
        t = time.time()
        print ( '%s: %.5fs' % (self.desc, t-self.start ))

EvalStats = namedtuple('EvalStats', ['TP', 'TN', 'FP', 'FN', 'P', 'N', 'TP1', 'TN1', 'FP1', 'FN1' ]  )    

class Score(object) :
    
    def __init__(self, predictions, parent=None) :
        # predictions = ( correct, prediction, exid )
        self.parent = parent
        self.TP = 0.0
        self.FP = 0.0
        self.FN = 0.0
        self.TN = 0.0
        self.P = 0.0
        self.N = 0.0
        self.covered = []
        self.not_covered = []
        for p, ph, example_id in predictions :
            n = 1-p
            nh = 1-ph
            tp = min(p,ph)
            tn = min(n,nh)
            fp = n - tn
            fn = p - tp
            self.TP += tp
            self.TN += tn
            self.FP += fp
            self.FN += fn
            self.P += p
            self.N += n
            if ph == 0.0 :
                self.not_covered.append( (p, ph, example_id) )
            else :
                self.covered.append( (p, ph, example_id) ) 
                
    TP1 = property(lambda s : s[-1].TP )
    FP1 = property(lambda s : s[-1].FP )
    TN1 = property(lambda s : s[-1].TN )
    FN1 = property(lambda s : s[-1].FN )
        
    def __getitem__(self, index) :
        if index == 0 :
            return self
        elif index < 0 and self.parent :
            return self.parent[index+1]
        else :
            raise IndexError()
            
    def __str__(self) :
        return '<Score: %s %s %s %s>' % (self.TP, self.TN, self.FP, self.FN )

class RuleSet(object) :
    
    def __init__(self, RuleType, target, data) :
        self.target = target
        self.rules = []
        self.data = data
        self.Rule = RuleType
        
        examples = []
        for ex in self.data :
            hP = self.data.find_fact(self.target.functor, self.data[ex])
            examples.append( ( hP, 0, ex ) )
        examples = sorted(examples)
        self.score = Score(examples)
        self.P = self.score.P
        self.N = self.score.N
                                
    def getTP(self) :
        return self.score.TP
    
    def getTN(self) :
        return self.score.TN
    
    def getFP(self) :
        return self.score.FP
    
    def getFN(self) :
        return self.score.FN

    def getTP1(self) :
        return self.score[-1].TP
    
    def getTN1(self) :
        return self.score[-1].TN
    
    def getFP1(self) :
        return self.score[-1].FP
    
    def getFN1(self) :
        return self.score[-1].FN

        
    TP = property(getTP)    
    TN = property(getTN)
    FP = property(getFP)
    FN = property(getFN)
    
    COVERED = property(lambda s : s.score.covered)
    NOT_COVERED = property(lambda s : s.score.not_covered)

    TP1 = property(getTP1)    
    TN1 = property(getTN1)
    FP1 = property(getFP1)
    FN1 = property(getFN1)

    def refine(self, update=False) :
        return self.rules[-1].refine(self.data, update)
        
    def pushRule(self, rule=None) :
        if rule == None : rule = self.newRule()
        
        evaluated = [] # discard COVERED examples
        for p, ph, example in self.NOT_COVERED :
            h, b = rule.evaluate(self.data, self.data[example])
            evaluated.append( (p, b, example ) )
            
        self.score = Score( evaluated, self.score )
        self.rules.append(rule)

    def popRule(self) :
        self.score = self.score[-1]
        self.rules.pop(-1)
        
    def testLiteral(self, literal) :
        current_rule = self.rules[-1]
        
        new_rule = current_rule + literal
        # can only move examples from self.XXX[-1] to self.XXX[0]
        
        evaluated = self.NOT_COVERED[:]
        for p, ph, ex in self.COVERED :
            h, b = new_rule.evaluate(self.data, self.data[ex])
            evaluated.append( (p, b, ex ) )
        return Score(evaluated, self.score )
        
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

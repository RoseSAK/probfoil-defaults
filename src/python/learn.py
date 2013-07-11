#! /usr/bin/env python3

#from __future__ import print_function

from collections import namedtuple
import time, sys

BEAM_SIZE = 5

# def localScore(rs) :
#     m = 10    
#     m_estimate_c = rs.score.m_estimate(m)
#     m_estimate_p = rs.score[-1].m_estimate(m)
#     return m_estimate_c - m_estimate_p
    
def globalScore(rs) :
    return rs.score.accuracy()
    
def learn(H) :
    score_H = globalScore(H)
    while True :
        rule = best_clause( H )
        H.pushRule(rule)        
        score_nH = globalScore(H)
        
        print >> sys.stderr, 'BETTER?', score_H, score_nH
        if (score_H >= score_nH) :
            H.popRule()
            break
        else :
            score_H = score_nH
    return H

def best_clause( H, beam_size = BEAM_SIZE ) :
    beam = Beam(beam_size)    
    rule = H.newRule()
    refinements = [ (0,r) for r in rule.refine(H.data) ]
    beam.push( rule, refinements , None )

    while beam.has_active() :
        new_beam = Beam(beam_size)
        for score, rule, refs in beam :
            print >> sys.stderr, "REFINING", score, rule, refs, '\n\n'
            new_beam.push( rule, None, score )
            H.pushRule(rule)
            new_refs = update_refinements(H, refs)
            print >> sys.stderr, '>>>>', new_refs
            for i, score_ref in enumerate(new_refs) :
                new_score, ref = score_ref
                if new_score.FP == 0 and new_score.FN == 0 :
                    return rule + ref
                if new_score <= score or not new_beam.push( rule + ref, new_refs[i+1:], new_score ) : # was new_refs[i+1:]
                    break  # current ref does not have high enough score -> next ones will also fail
            # print >> sys.stderr, new_beam
            H.popRule()
        beam = new_beam
        print >> sys.stderr, beam
        
    print >> sys.stderr, 'FOUND RULE',beam.content[0][1]
    return beam.content[0][1]

class Beam(object) :
    
    def __init__(self, size) :
        self.size = size
        self.content = []
       
    def __iter__(self) :
        return iter(self.content)
        
    def pick_best(self, rec1, rec2) :
        if len(rec1[1]) > len(rec2[1]) :
            return rec2
        elif len(rec1[1]) < len(rec2[1]) :
            return rec1
        elif rec1[1].countNegated() < rec2[1].countNegated() :
            return rec1
        elif rec2[1].countNegated() < rec1[1].countNegated() :
            return rec2
        else :
            return rec1

        
    def push(self, obj, active, score) :
        if len(self.content) == self.size and score < self.content[-1][0] : return False
        
        is_last = True
        
        p = len(self.content) - 1
        self.content.append( (score, obj, active) )
        while p >= 0 and self.content[p][0] < self.content[p+1][0] :
            self.content[p], self.content[p+1] = self.content[p+1], self.content[p] # swap elements
            p = p - 1
            is_last = False
        
        if self.content[p][0] == self.content[p+1][0] :
            self.content[p+1] = self.pick_best(self.content[p], self.content[p+1])
#            print >> sys.stderr, 'REMOVED', self.content[p][1], self.content[p+1][1], '==>', self.content[p+1][1]
            self.content = self.content[:p] + self.content[p+1:]    # remove p

            
        
        popped_last = False
        while len(self.content) > self.size :
            self.content.pop(-1)
            popped_last = True
            
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
            res += str(c) + ': ' + str(s) +  ' | ' + str(r) + '\n'
        res += '>>>'
        return res
        
def update_refinements(H, refine) :
    if refine == None :
        return []
    
    literals = []
    
    new_refine = [ (0,r) for r in H.refine(update=True) ]
    
    if new_refine : print >> sys.stderr, 'NEW REFINEMENTS', new_refine
    
    refine += new_refine
#    
#    refine = set(refine)
#    refine = new_refine
    
    for s, lit in refine :
        
        new_score = H.testLiteral(lit)
        
        # H.score = score of hypothesis with rule but without literal
        # H.score[-1] = new_score[-1] = score of hypothesis without rule
        # new_score = score of hypothesis with rule and literal

        if new_score.TN > H.score.TN and new_score.TP > new_score[-1].TP :
            # true negatives should be up because of this LITERAL (adding literals increases this)
            # true positives should be up because of this RULE (adding literals decreases this)
            
           # current_score = score_func(new_score)
            print >> sys.stderr, 'accepted', H.rules[-1], lit, new_score.TN - H.score.TN , new_score.TP - new_score[-1].TP, new_score, new_score[-1]
            literals.append( (new_score, lit) )
        else :
            # literal doesn't cover true positives or it doesn't eliminate false positives
            print >> sys.stderr, 'rejected', H.rules[-1], lit, new_score.TN - H.score.TN , new_score.TP - new_score[-1].TP, new_score, new_score[-1]
    return list(reversed(sorted(literals)))        
        
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
        
        if parent :
            self.TP = parent.TP
            self.FP = parent.FP
        
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
        self.covered = sorted(self.covered)
        self.not_covered = sorted(self.not_covered)
                        
    def __getitem__(self, index) :
        if index == 0 :
            return self
        elif index < 0 and self.parent :
            return self.parent[index+1]
        else :
            raise IndexError()
            
#    score = property(lambda s : s)
            
    def value(self) :
        m = 10    
        m_estimate_c = self.m_estimate(m)
        if self.parent :
            m_estimate_p = self[-1].m_estimate(m)
            return m_estimate_c - m_estimate_p
        else :
            return None
            
    def m_estimate(self, m) :
        return (self.TP + m * (self.P / (self.N + self.P))) / (self.TP + self.FP + m) 
        
    def accuracy(self) :
        return (self.TP + self.TN ) / (self.TP + self.TN + self.FP + self.FN)
            
    def __str__(self) :
        return '<Score: %s %s %s %s | %s>' % (self.TP, self.TN, self.FP, self.FN, self.value() )
        
    def __repr__(self) :
        return str(self.value())
        
    def __cmp__(self, other) :
        if other :
            return cmp(self.value(), other.value())
        else :
            return 1

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
                                
    TP = property(lambda s : s.score.TP)    
    TN = property(lambda s : s.score.TN)
    FP = property(lambda s : s.score.FP)
    FN = property(lambda s : s.score.FN)
    
    COVERED = property(lambda s : s.score.covered)
    NOT_COVERED = property(lambda s : s.score.not_covered)

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
        return Score(evaluated, self.score[-1] )
        
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

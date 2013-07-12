#! /usr/bin/env python3

from __future__ import print_function

from collections import namedtuple
import time, sys

from util import Timer, Log, Beam

SettingsType = namedtuple('Settings', ['BEAM_SIZE', 'M_ESTIMATE_M', 'EQUIV_CHECK'] )
SETTINGS = SettingsType(5,10,True)

def learn(H) :
    score_H = H.score.globalScore
    while score_H < 1.0 :
      with Log('learn_rule'):
        rule = best_clause( H )
        H.pushRule(rule)        
        score_nH = H.score.globalScore

        Log('rule_found', rule=rule, score=H.score).logline()
        Log('stopping_criterium', old_score=score_H, new_score=score_nH).logline()

        if (score_H >= score_nH) :
            H.popRule()
            break
        else :
            score_H = score_nH
    return H

def best_clause( H ) :
    beam = Beam(SETTINGS.BEAM_SIZE, not SETTINGS.EQUIV_CHECK)    
    rule = H.newRule()
    refinements = [ (0,r) for r in rule.refine(H.data) ]
    beam.push( rule, refinements , None )

    it=0
    while beam.has_active() :
      it += 1
      with Log('iteration', index=it) :
        new_beam = beam.create()
        for score, rule, refs in beam :
          with Log('refining', rule=rule, score=score) :
            new_beam.push( rule, None, score )
            H.pushRule(rule)
            new_refs = update_refinements(H, refs)
            H.popRule()
            for i, score_ref in enumerate(new_refs) :
                new_score, ref = score_ref
                if new_score.FP == 0 and new_score.FN == 0 :
                    return rule + ref   # we found a rule with maximal score => no better rule can be found
                if new_score <= score or not new_beam.push( rule + ref, new_refs[i+1:], new_score ) : # was new_refs[i+1:]
                    break  # current ref does not have high enough score -> next ones will also fail
        beam = new_beam
        
        with Log('beam') as l :
            if l.file :
                print(beam.toXML(), file=l.file)
     
    return beam.content[0][1]

def pick_best_literal(rec1, rec2) :
    if rec1[1].isNegated() :
        return rec2, rec1
    else :
        return rec1, rec2
    
def remove_dups(literals) :
    
    prev_score = None
    result = []
    for score, lit in literals :
        if result and result[-1][0] == score :
            best, worst = pick_best_literal(result[-1], (score,lit))
            result[-1] = best
            Log('equivalent_literal', best=best[1], worst=worst[1]).logline()
        else :
            result.append( (score,lit) )
    return result

def update_refinements(H, refine) :
    if refine == None :
        return []
    
    literals = []
    
    new_refine = [ (0,r) for r in H.refine(update=True) ]
    
#    if new_refine : 
#        Log('newrefine', refinements=new_refine).logline()

    Log('refinements', current='|'.join(map(lambda s : str(s[1]), refine)), new='|'.join(map(lambda s : str(s[1]), new_refine))).logline()

    
    refine += new_refine
#    
#    refine = set(refine)
#    refine = new_refine
    
    new_scores = H.testLiterals(refine)
    
    for new_score, lit in new_scores :        
        # H.score = score of hypothesis with rule but without literal
        # H.score[-1] = new_score[-1] = score of hypothesis without rule
        # new_score = score of hypothesis with rule and literal

        if new_score.TN > H.score.TN and new_score.TP > new_score[-1].TP :
            # true negatives should be up because of this LITERAL (adding literals increases this)
            # true positives should be up because of this RULE (adding literals decreases this)
           # current_score = score_func(new_score)
            s = 'accepted'          
            literals.append( (new_score, lit) )
        else :
            # literal doesn't cover true positives or it doesn't eliminate false positives
            # TODO what if it introduces a new variable?
            s = 'rejected'
        Log(s, literal=lit, tn_change=new_score.TN - H.score.TN , tp_change=new_score.TP - new_score[-1].TP, score=new_score ).logline()

    result = list(reversed(sorted(literals)))
    
    if SETTINGS.EQUIV_CHECK :
        result = remove_dups(result)
    
    Log('update_refinements', result=result).logline()
    return result
        

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
            
    def _calc_local(self) :
        m = SETTINGS.M_ESTIMATE_M  
        m_estimate_c = self.m_estimate(m)
        if self.parent :
            m_estimate_p = self[-1].m_estimate(m)
            return m_estimate_c - m_estimate_p
        else :
            return None
    
    def _calc_global(self) : 
        return self.accuracy()
            
    globalScore = property(lambda s : s._calc_global() )
    localScore = property(lambda s : s._calc_local() )
            
    def m_estimate(self, m) :
        return (self.TP + m * (self.P / (self.N + self.P))) / (self.TP + self.FP + m) 
        
    def accuracy(self) :
        return (self.TP + self.TN ) / (self.TP + self.TN + self.FP + self.FN)
            
    def __str__(self) :
        ls = self.localScore
        if ls == None :
            return '%s %s %s %s None' % (self.TP, self.TN, self.FP, self.FN )
        else :
            return '%s %s %s %s %.4f' % (self.TP, self.TN, self.FP, self.FN, self.localScore )
        
    def __repr__(self) :
        return str(self.localScore)
        
    def __eq__(self, other) :
        return self.covered == other.covered
        
    def __cmp__(self, other) :
        if other :
            return cmp(self.localScore, other.localScore)
        else :
            return 1
            
    def extend(self, evaluated) :
        return Score(evaluated, self)

class RuleSet(object) :
    
    def __init__(self, RuleType, ScoreType, target, data) :
        self.target = target
        self.rules = []
        self.data = data
        self.Rule = RuleType
        
        examples = []
        for ex in self.data :
            hP = self.data.find_fact(self.target.functor, self.data[ex])
            examples.append( ( hP, 0, ex ) )
        examples = sorted(examples)
        self.score = ScoreType(examples)
        self.P = self.score.P
        self.N = self.score.N
                                    
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
            
        self.rules.append(rule)
        self.score = self.score.extend(evaluated)

    def popRule(self) :
        self.score = self.score[-1]
        self.rules.pop(-1)        
        
    def testLiterals(self, literals) :
        return ( (self.testLiteral(lit), lit) for s,lit in literals )

    def testLiteral(self, literal) :
        current_rule = self.rules[-1]
        
        new_rule = current_rule + literal

        evaluated = self.NOT_COVERED[:]
        for p, ph, ex in self.COVERED :
            h, b = new_rule.evaluate(self.data, self.data[ex])
            evaluated.append( (p, b, ex ) )
        return self.score[-1].extend(evaluated)
                
    def newRule(self) :
        return self.Rule(self.target)
        
    def __str__(self) :
        return '\n'.join(map(str,self.rules))        


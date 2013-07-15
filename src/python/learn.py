#! /usr/bin/env python3

from __future__ import print_function

from collections import namedtuple
import time, sys

from util import Timer, Log, Beam

SettingsType = namedtuple('Settings', ['BEAM_SIZE', 'M_ESTIMATE_M', 'EQUIV_CHECK'] )
SETTINGS = SettingsType(5,10,False)     # TODO fix equiv_check for python 3

def learn(H) :
    """Core FOIL learning algorithm.
    
    H - initial hypothesis (Hypothesis)
    """
    
    # Store current hypothesis score for use in stopping criterion
    score_H = H.score.globalScore
    
    # Find clauses as long as stopping criterion is not met or until maximal score (1.0) is reached.
    while score_H < 1.0 :   # this test is not required for correctness (alternative: while True)
       
      with Log('learn_rule'):
            
        # Find best clause that refines this hypothesis
        rule, score = best_clause( H )

        # Add the clause to the hypothesis (updates score)
        H.pushRule(rule, score)        

        # Retrieve the score for the new hypothesis
        score_nH = H.score.globalScore

        # Log progress
        Log('rule_found', rule=rule, score=H.score).logline()
        Log('stopping_criterion', old_score=score_H, new_score=score_nH, full_score=H.score).logline()

        # Check stopping criterion
        if (score_H >= score_nH) :
            # Clause does not improve hypothesis => remove it and stop
            H.popRule()
            break
        else :
            # Clause improves hypothesis => continue
            score_H = score_nH
    return H

def best_clause( H ) :
    """Find the best clause for this hypothesis."""
    
    # We use beam search
    beam = Beam(SETTINGS.BEAM_SIZE, not SETTINGS.EQUIV_CHECK)    
    
    # Create the new rule ( target <- )
    rule = H.newRule()
    
    # Calculate initial set of refinements
    refinements = [ (0,r) for r in rule.refine(H.data) ]
    
    # Add clause to beam (with empty score)
    beam.push( rule, refinements , H.default_score() )

    it=0

    # While there are untested refinements in the beam.
    while beam.has_active() :
      it += 1
      with Log('iteration', index=it) :
          
        # Create the next beam
        new_beam = beam.create()
        for score, rule, refs in beam :
          with Log('refining', rule=rule, score=score) :
              
            # Add current rule to beam and mark it as tested (refinements=None)
            new_beam.push( rule, None, score )
            
            # Add the current rule to the hypothesis (updates scores)
            H.pushRule(rule, score)
            
            # Update scores of available refinements and add new refinements if a new variable was introduced.
            new_refs = update_refinements(H, refs)
            
            # Remove rule from hypothesis (resets scores)
            H.popRule()
            
            # Add rules to new beam (new_refs are ordered by score, descending)
            for i, score_ref in enumerate(new_refs) :
                new_score, ref = score_ref
                if new_score.FP == 0 and new_score.FN == 0 :
                    return rule + ref, new_score   # we found a rule with maximal score => no better rule can be found
                if (not score == None and new_score <= score) or not new_beam.push( rule + ref, new_refs[i+1:], new_score ) : # was new_refs[i+1:]
                    break  # current ref does not have high enough score -> next ones will also fail

        # Use new beam in next iteration
        beam = new_beam
        
        # Write beam to log
        Log('beam').logXML(beam.toXML())

    # Return head of beam.
    return beam.content[0][1], beam.content[0][0]

def update_refinements(H, refine) :
    if refine == None :
        return []
    
    # Calculate new refinements in case a variable was added by the previous literal
    new_refine = [ (0,r) for r in H.refine(update=True) ]
    
    # Log
    Log('refinements', current='|'.join(map(lambda s : str(s[1]), refine)), new='|'.join(map(lambda s : str(s[1]), new_refine))).logline()

    # Add new refinements
    refine += new_refine
    
    # Update scores for all literals in batch
    new_scores = list(reversed(sorted(H.testLiterals([ lt for sc, lt in refine ]))))
    
    Log('new_scores', scores=new_scores).logline()
    
    # Reject / accept literals based on a local stopping criterion
    result = []
    for new_score, lit in new_scores :        
        # H.score = score of hypothesis with rule but without literal
        # H.score.parent = new_score.parent = score of hypothesis without rule
        # new_score = score of hypothesis with rule and literal

        if new_score.TN > H.score.TN and new_score.TP > new_score.parent.TP :
            # true negatives should be up because of this LITERAL (adding literals increases this)
            # true positives should be up because of this RULE (adding literals decreases this)
            # current_score = score_func(new_score)
            Log('accepted', literal=lit, tn_change=new_score.TN - H.score.TN , tp_change=new_score.TP - new_score.parent.TP, score=new_score ).logline()
            
            # Check whether this literal is equivalent to previous (covers exactly the same examples with the same score)
            if SETTINGS.EQUIV_CHECK and result and result[-1][0] == new_score :
                worst, best = sorted( (result[-1][1], lit) )
                result[-1] = new_score, best
                Log('equivalent_literal', best=best, worst=worst).logline()
            else :
                result.append( (new_score, lit) )
        else :
            # literal doesn't cover true positives or it doesn't eliminate false positives
            # TODO what if it introduces a new variable?
            Log('rejected', literal=lit, tn_change=new_score.TN - H.score.TN , tp_change=new_score.TP - new_score.parent.TP, score=new_score ).logline()
        
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
        
        # Examples covered by previous clauses are eliminated,
        # so we should keep statistics of positive predictions of these previous rules.
        if parent :
            self.TP = parent.TP
            self.FP = parent.FP
            self.P = parent.P
            self.N = parent.N
        
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
            if not self.parent :
                self.P += p
                self.N += n
            if ph == 0.0 :
                self.not_covered.append( (p, ph, example_id) )
            else :
                self.covered.append( (p, ph, example_id) ) 

        # Sort to make scores fully comparable
        if SETTINGS.EQUIV_CHECK :
            self.covered = sorted(self.covered)
            self.not_covered = sorted(self.not_covered)
                                    
    def _calc_local(self) :
        m = SETTINGS.M_ESTIMATE_M  
        m_estimate_c = self.m_estimate(m)
        if self.parent :
            m_estimate_p = self.parent.m_estimate(m)
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
        if other == None :
            return False
        else :
            return self.localScore == other.localScore # and self.covered == other.covered
        
    def __lt__(self, other) :
        if other == None :
            return False
        else :
            return self.localScore < other.localScore

    def __le__(self, other) :
        if other == None :
            return False
        else :
            return self.localScore <= other.localScore

        
    # def __cmp__(self, other) :
    #     if other :
    #         return cmp(self.localScore, other.localScore)
    #     else :
    #         return 1
            
    def extend(self, evaluated) :
        return Score(evaluated, self)

class Hypothesis(object) :
    
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
        
    def default_score(self) :
        examples = []
        for ex in self.data :
            hP = self.data.find_fact(self.target.functor, self.data[ex])
            examples.append( ( hP, 1, ex ) )
        examples = sorted(examples)
        return self.score.extend(examples)
                                    
    COVERED = property(lambda s : s.score.covered)
    NOT_COVERED = property(lambda s : s.score.not_covered)

    def refine(self, update=False) :
        return self.rules[-1].refine(self.data, update)
        
    def pushRule(self, rule, score) :
        # if rule == None : rule = self.newRule()
        # 
        # evaluated = [] # discard COVERED examples
        # for p, ph, example in self.NOT_COVERED :
        #     b = rule.evaluate(self.data, self.data[example])
        #     evaluated.append( (p, b, example ) )
        #     
        #self.score = self.score.extend(evaluated)
        self.rules.append(rule)
        self.score = score

    def popRule(self) :
        self.score = self.score.parent
        self.rules.pop(-1)        
        
    def testLiterals(self, literals) :
        current_rule = self.rules[-1]
        
        scores, _dummy, ex_ids = zip(*self.COVERED)
        
        evaluated = current_rule.evaluate_extensions(self.data, [ self.data[ex] for ex in ex_ids ], literals)
        
        from itertools import chain
        for lit, new_scores in evaluated :
            yield self.score.parent.extend( chain(self.NOT_COVERED, zip(scores, new_scores, ex_ids)) ), lit        
        
    def testLiteral(self, literal) :
        current_rule = self.rules[-1]
        
        new_rule = current_rule + literal

        evaluated = self.NOT_COVERED[:]
        for p, ph, ex in self.COVERED :
            b = new_rule.evaluate(self.data, self.data[ex])
            evaluated.append( (p, b, ex ) )
        return self.score.parent.extend(evaluated)
                
    def newRule(self) :
        return self.Rule(self.target)
        
    def __str__(self) :
        return '\n'.join(map(str,self.rules))        


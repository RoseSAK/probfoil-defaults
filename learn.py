# Copyright (C) 2013 Anton Dries (anton.dries@cs.kuleuven.be)
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2.1 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import math
from util import Log, Beam, Timer
from language import RuleHead
import time

# Helper function for transforming significance p-value into ChiSquare decision value.
chi2_cdf = lambda x : math.erf(math.sqrt(x/2))
def calc_significance(s, low=0.0, high=100.0, precision=1e-8) :
    v = (low+high)/2
    r = chi2_cdf(v)
    if -precision < r - s < precision :
        return v
    elif r > s :
        return calc_significance(s, low, v)
    else :
        return calc_significance(s, v, high)


class LearningProblem(object) :
    """Base class for FOIL learning algorithm."""
    
    def __init__(self, language, knowledge, beam_size=5, significance_p_value=0.99, verbose=False, minrules=0, maxrules=-1, pack_queries=True, use_recall=False, **other_args ) :
        self.language = language
        self.knowledge = knowledge
        
        self.BEAM_SIZE = beam_size
        self.VERBOSE = verbose
        self.SIGNIFICANCE = calc_significance(significance_p_value)
        if self.VERBOSE > 2 : print ('Significance set to:', self.SIGNIFICANCE )
        
        self.MINRULES = minrules
        self.MAXRULES = maxrules
        self.PACK_QUERIES = pack_queries
        self.USE_RECALL = use_recall
        
    def calculateScore(self, rule) :
        raise NotImplementedError('calculateScore')

    def learn(self, H) :
        """Core FOIL learning algorithm.
    
        H - initial hypothesis (Rule (e.g. RootRule))
        """
    
        rule_count = 0
        
        # Find clauses as long as stopping criterion is not met or until maximal score (1.0) is reached.
        while H.globalScore < 1.0 :   # this test is not required for correctness (alternative: while True)
       
          with Log('learn_rule', _timer=True):
            
            # Find best clause that refines this hypothesis
            new_H = self.best_clause( H )
            
            # Prune rule 
            #   Explanation: in case of equal score the search procedure prefers rules with more variables
            #                once the search is over we prefer shorter rules
            while new_H.parent and new_H.parent.localScore >= new_H.localScore :
                new_H = new_H.parent
        
            if self.VERBOSE > 0 : print ('RULE FOUND:', new_H, '%.5f' % new_H.globalScore)
        
            # Log progress
            with Log('rule_found', rule=new_H, score=new_H.score) : 
                pass
            with Log('stopping_criterion', old_score=H.globalScore, new_score=new_H.globalScore, full_score=new_H.score, sign=new_H.significance) : 
                pass
        
            # Check significance level 
            if new_H.significance < self.SIGNIFICANCE :
                # Clause not significant => STOP
                break
            
            # Check stopping criterion
            if self.MINRULES > rule_count or new_H.globalScore > H.globalScore :
                # Clause improves hypothesis => continue
                H = new_H
                rule_count += 1
            else :
                # Clause does not improve hypothesis => remove it and stop
                break
            if self.MAXRULES > 0 and rule_count >= self.MAXRULES :
                # Maximal number of rules reached => stop
                break
        return H
        
    def best_clause(self, current_rule ) :
        """Find the best clause for this hypothesis."""
    
        # We use beam search; initialize beam
        beam = Beam(self.BEAM_SIZE)    
    
        # Create the new rule ( target <- )
        init_rule = RuleHead(previous=current_rule)
    
        # Calculate initial set of refinements
        with Timer(category='refine') :
            refinements = list(init_rule.refine())
    
        # Add clause to beam (with empty score)
        beam.push( init_rule, refinements )
        
        # Keep track of best score so far.
        best_rule = init_rule
        try :
            # While there are untested refinements in the beam.
            while beam.has_active() :
              with Log('iteration', _timer=True) :
          
                # Create the next beam
                new_beam = beam.create()
        
                # Run through old beam and process its content
                for old_rule, refs in beam :
                
                  with Log('refining', rule=old_rule, score=old_rule.score, localScore=old_rule.localScore, _timer=True) :
                    if best_rule != None and old_rule.localScoreMax < best_rule.localScore :
                        # Prune this rule, it can never reach the top of the beam.
                        with Log('abort', reason="maxscore", bestscore=best_rule.localScore, maxscore=old_rule.localScoreMax) : pass
                        continue 
            
                    # Add current rule to beam and mark it as tested (refinements=None)
                    # new_beam.push( old_rule, None )
                        
                    # Update scores of available refinements and add new refinements if a new variable was introduced.
                    best_score = None
                    if best_rule : best_score = best_rule.localScore
                    new_rules = self.update_refinements(old_rule, refs, best_score)
            
                    # Extract refinement literals
                    new_refs = [ r.literal for r in new_rules ]
                            
                    # Add rules to new beam (new_refs are ordered by score, descending)
                    for i, new_rule in enumerate(new_rules) :
                        
                        # Update best score
                        current_score = new_rule.localScore
                        if best_rule == None or current_score > best_rule.localScore :
                            best_rule = new_rule
                
                        if self.VERBOSE > 2 : print ( '%s %s %.5f %.5f %.5f' % (new_rule,new_rule.score, new_rule.localScore, new_rule.significance, new_rule.max_significance) )
                
                        # Early stopping
                        if new_rule.score.FP == current_rule.score.FP and new_rule.score.FN == 0 :
                           return new_rule   # we found a rule with maximal score => no better rule can be found
                        else :
                            if new_rule.score.FP == current_rule.score.FP :
                                # There are no false positives => we cannot get better by extending this rule
                                next_refs = None
                            else :
                                # Default case, only allow worse extensions (symmetry breaking)
                                next_refs = new_refs[i+1:]
                                
                            # Attempt to add rule to beam
                            if not new_beam.push( new_rule , next_refs ) : 
                                break  # current ref does not have high enough score -> next ones will also fail
            
                # Use new beam in next iteration
                beam = new_beam
        
                # Write beam to log
                with Log('beam', _child=beam.toXML()) : pass
        except KeyboardInterrupt :
            # Allow interrupting search process, algorithm will continue with current best rule.
            with Log('INTERRUPT') : pass
            print ("INTERRUPT")
        
        # Return head of beam.
        return best_rule

    def update_refinements(self, rule, refine, best_score) :

        if refine == None :
            # No refinements available
            return []
            
        # Calculate new refinements in case a variable was added by the previous literal    
        with Timer(category='refine') :
            new_refine = list(rule.refine(update=True))
            
            # Add new refinements
            refine += new_refine
                
        if self.VERBOSE > 2 : print('Evaluating %s refinements...' % len(refine) )
            
        with Log('new_refine', new=new_refine, all=refine) : pass
        
        # Update scores for all literals in batch
        refine = sorted((rule + ref for ref in refine), reverse = True)
        
        # Reject / accept literals based on a local stopping criterion
        result = []
        for r in refine :
        
            prev_score = r.previous.score
            parent_score = r.parent.score
            new_score = r.score
            
            if new_score.TP <= prev_score.TP + 1e-10 :
                # Rule doesn't cover any true positive examples => it's useless
                with Log('rejected', reason="TP", literal=r.literal, score=r.score, localScore=r.localScore ) : pass
            elif r.max_significance < self.SIGNIFICANCE :
                # Rule cannot reach required significance => it's useless
                with Log('rejected', reason="s", literal=r.literal, score=r.score, max_significance=r.max_significance ) : pass
            elif not r._new_vars and r.samePredictions(r.parent) :
                # Predictions are indentical and new rule does not introduce a new variable
                with Log('rejected', reason="no improvement", literal=r.literal, score=r.score ) : pass
            elif best_score != None and r.localScoreMax <= best_score and r.localScore < best_score :
                # This rule and any of its extensions can be better than the currently best rule
                with Log('rejected', reason="maxscore", literal=r.literal, score=r.score, localscore=r.localScore, maxscore=r.localScoreMax ) : pass
#            elif r.score.TP <= r.parent.score.TP and r.score.TN == r.parent.score.TN and not r._new_vars :
#                 # DISABLED: combination with symmetry breaking on refinements might eliminate options
#               # Rule covers less TP and same TN and does not introduce new variable
#                with Log('rejected', reason="worse", literal=r.literal, score=r.score ) : pass
            else :
                # Accept the extension and add it to the output
                with Log('accepted', literal=r.literal, score=r.score, localScore=r.localScore, maxScore=r.localScoreMax) : pass
                result.append( r )
                # Update best score
                if best_score == None or r.localScore > best_score : best_score = r.localScore
    
        return result
    
class ProbFOIL1(LearningProblem) :
    
    def __init__(self, *args,  **kwdargs) :
        super(ProbFOIL1,self).__init__(*args, **kwdargs)
        self.M_ESTIMATE_M = kwdargs.get('m_estimate_m',10)
    
    def calculateScore(self, rule) :
        if not rule.previous :
            return PF1Score(rule.score_correct, rule.getScorePredict(), self.M_ESTIMATE_M, 0.0, 0.0)
        else :
            return PF1Score(rule.score_correct, rule.getScorePredict(), self.M_ESTIMATE_M, rule.previous.score.TP, rule.previous.score.FP)

class ProbFOIL2(LearningProblem) :
    
    def __init__(self, *args, **kwdargs) :
        super(ProbFOIL2,self).__init__(*args, **kwdargs)
        self.M_ESTIMATE_M = kwdargs.get('m_estimate_m',10)
    
    def calculateScore(self, rule) :
        if not rule.previous :
            return PF1Score(rule.score_correct, rule.getScorePredict(), self.M_ESTIMATE_M, 0.0, 0.0)
        else :
            previous_prediction = rule.previous.getScorePredict()            
            result = PF2Score(rule.score_correct, rule.getScorePredict(), previous_prediction, self.M_ESTIMATE_M)
            return result

class PFScore(object) :
    
    def __init__(self) :
        pass
        
    # Calculate actual significance
    def calculate_significance(self, calc_max=False) :
        
        pTP = self.pTP
        pFP = self.pFP
        
        pP = pTP
        pN = pFP
        pM = pP + pN

        s = self
        
        if calc_max :
            sTP = s.maxTP - pTP
            sFP = 0
        else :
            sTP = s.TP - pTP
            sFP = s.FP - pFP
            
        sP = s.P # - pP
        sN = s.N # - pN
        sM = sP + sN

        C = sTP + sFP           # max: C == sTP (sFP == 0)
        if C == 0 : return 0
            
        p_pos_c = sTP / C       # max: p_pos_c == 1 
        p_neg_c = 1 - p_pos_c   # max: p_neg_c == 0
        
        p_pos = sP / sM
        p_neg = sN / sM
        
        pos_log = math.log(p_pos_c/p_pos) if p_pos_c > 0 else 0     # max: pos_log = -log(sP / sM)
        neg_log = math.log(p_neg_c/p_neg) if p_neg_c > 0 else 0     # max: neg_log = 0
        
        l = 2*C * (p_pos_c * pos_log  + p_neg_c * neg_log  )        # max: 2 * sTP * -log(sP/sM)
        
        return l
    
        
    def accuracy(self) :
        M = self.P + self.N
        return (self.TP + self.TN ) / M
            
    def m_estimate(self) :
        return self._m_estimate_m(self.TP, self.FP)
        
    def m_estimate_max(self) :
        return self._m_estimate_m(self.maxTP, 0)
            
    def _m_estimate_m(self, TP, FP) :
        return (TP + self.mPNP) / (TP + FP + self.M_ESTIMATE_M) 

    def __str__(self) :
        return '%.3f %.3f %.3f %.3f' % (self.TP, self.TN, self.FP, self.FN )

    def recall(self) :
        return self.TP / (self.TP + self.FN)

    

class PF1Score(PFScore) :

    def __init__(self, correct, predict, m, pTP, pFP) :
        self.M_ESTIMATE_M = m
        self.max_x = 1
        self.TP = 0.0
        self.FP = 0.0
        self.FN = 0.0
        self.TN = 0.0
        self.P = 0.0
        self.N = 0.0
        self.pTP = pTP
        self.pFP = pFP
            
        for p, ph in zip(correct,predict) :
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
        self.maxTP = self.TP
        M = self.P + self.N
        self.mPNP = self.M_ESTIMATE_M  * ( self.P / M )
        self.localScore = self.m_estimate()
        self.localScoreMax = self.m_estimate_max()
        self.significance = self.calculate_significance()
        self.significance_max = self.calculate_significance(True)
                    
    def __str__(self) :
        return '%.3f %.3f %.3f %.3f' % (self.TP, self.TN, self.FP, self.FN )

class PF2Score(PFScore):
    
    def __init__(self, correct, predict, predict_prev, m) :
        self.M_ESTIMATE_M = m
        self.MIN_RULE_PROB = 0.01
        
        # Calculate the y values for which
        #values = [ (self._calc_y(p,l,u), p,l,u) for p,l,u in zip(correct, predict_prev, predict ) ]
        
        # There are four types of scores:
        #   Take dS = u - l
        #   - inactive: current rule is not adaptable (u == l)   => x is irrelevant
        #       dTP = 0
        #       dFP = 0
        #   - overestimating: current rule is an overestimate ( p < l )
        #       dTP = 0
        #       dFP = x*dS
        #   - underestimating: current rule is an underestimate ( p > u )
        #       dTP = x*dS
        #       dFP = 0
        #   - correctable: x can be adjusted to perfect prediction ( l < p < u )
        #       y = (p-l) / (u-l)
        #       if x < y :  (underestimate)
        #           dTP = x*dS
        #           dFP = 0
        #       elif y > x :  (overestimate)
        #           dTP = (p-l)
        #           dFP = (x*dS) - (p-l)
        #       else :  (correct)
        #           => p-l == x*dS
        #           => dTP = x*dS = p-l
        #           => dFP = 0
        #       Incremental computation:
        #           Take pl = p-l
        #           Take dS = u-l
        #           At position n, value[n] = x
        #               TP_x = x \sum_{i=n+1}^{M} dS_i + \sum_{i=1}^{n} pl_i 
        #               FP_x = x \sum_{i=1}^{n} dS_i - \sum_{i=1}^{n} pl_i
        #           We can compute this by maintaining the running sums:
        #               dS_running = \sum_{i=1}^{n} dS_i
        #               pl_running = \sum_{i=1}^{n} pl_i
        #           and the sum
        #               dS_total = \sum_{i=i}^{M} dS_i

        P = sum(correct)
        M = len(correct)
        N = M - P
        self.mPNP = self.M_ESTIMATE_M  * ( P / M )
        
        #print ('mPNP', self.mPNP)
        
        TP_previous = 0.0
        FP_previous = 0.0
        
        TP_base = 0.0
        FP_base = 0.0
        values = []
        
        dS_total = 0.0
        for p,l,u in zip(correct, predict_prev, predict ) :
                        
            assert( u > l - 1e-10)
        
            TP_previous += min(l,p)
            FP_previous += max(0,l-p)
            
            #print (p,l,u)
            
            dS = u - l
            if dS == 0 :    # inactive
                pass
            elif p < l :    # overestimate
                FP_base += dS
            elif p > u :    # underestimate
                TP_base += dS
            else : # correctable
                dS_total += dS
                y = (p-l) / (u-l)
                values.append( (y,p,l,u) )
                
        tau_l, tau_u, tau_p = 0.0, 0.0, 0.0     # Sum l_i
        sigma_l, sigma_u, sigma_p = 0.0, 0.0, 0.0 # Sum l_i: x>y 
        
        if values : 
            values = sorted(values)
        
            TP_x, FP_x, TN_x, FN_x = 0.0, 0.0, 0.0, 0.0
        
            max_score = None
            max_x = None
            max_score_details = None
        
            dS_running = 0.0
            pl_running = 0.0
            prev_y = None
            for y, p, l, u in values + [(None,0.0,0.0,0.0)] :
                if y == None  or (prev_y != None and y > prev_y) :
                    x = prev_y
                    
                    TP_x = pl_running + x * (dS_total - dS_running) + x * TP_base + TP_previous
                    FP_x = x * dS_running - pl_running + x * FP_base + FP_previous
                    
                    score_x = self._m_estimate_m(TP_x, FP_x)
                    #print (x, score_x, TP_x, FP_x, self._m_estimate_m(TP_x,0))
                    if x >= self.MIN_RULE_PROB and ( max_score == None or score_x > max_score ) :
                        TN_x = N - FP_x
                        FN_x = P - TP_x
                    
                        # TN_x = M - tau_p + sigma_p - (sigma_u - sigma_l) * x - sigma_l
                        # FN_x = tau_p - sigma_p - (tau_u - tau_l - sigma_u + sigma_l) * x - tau_l + sigma_l
                        max_score_details = (TP_x, TN_x, FP_x, FN_x)
                        max_score = score_x
                        max_x = x
                    
                prev_y = y
                
                pl_running += (p-l)
                dS_running += (u-l)

            
                
            if max_x == None or max_x > 1 - 1e-5 :
                x = 1
                TP_x = pl_running + x * (dS_total - dS_running) + x * TP_base + TP_previous
                FP_x = x * dS_running - pl_running + FP_previous
            
                TN_x = N - FP_x
                FN_x = P - TP_x
                
                max_score_details = (TP_x, TN_x, FP_x, FN_x)
                max_score = score_x
                max_x = x
                
        else :
            max_x = 1.0
            TP_x = TP_base + TP_previous
            FP_x = FP_base + FP_previous
            TN_x = N - FP_x
            FN_x = P - TP_x
            score_x = self._m_estimate_m(TP_x, FP_x)
            max_score_details = (TP_x, TN_x, FP_x, FN_x)
            max_score = score_x
            
        self.max_s = max_score
        self.max_x = max_x
        self.TP, self.TN, self.FP, self.FN = max_score_details
        self.P = P
        self.N = M-P
        
        self.maxTP = TP_x
        self.localScore = self.m_estimate()
        self.localScoreMax = self.m_estimate_max()
        self.pTP = TP_previous
        self.pFP = FP_previous
        self.significance = self.calculate_significance()
        self.significance_max = self.calculate_significance(True)

def test( correctF, predict_prevF, predictF ) :
    
    with open(correctF) as f :
        correct = map(float,f.readlines())

    with open(predict_prevF) as f :
        predict_prev = map(float,f.readlines())

    with open(predictF) as f :
        predict = map(float,f.readlines())

    s = PF2Score(correct, predict, predict_prev, 10)
    
    print (s, s.max_x)

if __name__ == '__main__' :
    import sys
    test(*sys.argv[1:])
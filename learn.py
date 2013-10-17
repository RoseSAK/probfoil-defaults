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
    
    def __init__(self, language, knowledge, beam_size=5, significance_p_value=0.99, verbose=False, **other_args ) :
        self.language = language
        self.knowledge = knowledge
        
        self.BEAM_SIZE = beam_size
        self.VERBOSE = verbose
        self.SIGNIFICANCE = calc_significance(significance_p_value)
        
    def calculateScore(self, rule) :
        raise NotImplementedError('calculateScore')

    def learn(self, H) :
        """Core FOIL learning algorithm.
    
        H - initial hypothesis (Rule (e.g. RootRule))
        """
    
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
        
            if self.VERBOSE > 1 : print ('RULE FOUND:', new_H, new_H.globalScore)
        
            # Log progress
            with Log('rule_found', rule=new_H, score=new_H.score) : 
                pass
            with Log('stopping_criterion', old_score=H.globalScore, new_score=new_H.globalScore, full_score=new_H.score) : 
                pass
        
            # TODO check significance level?
            
            # Check stopping criterion
            if (H.globalScore >= new_H.globalScore) :
                # Clause does not improve hypothesis => remove it and stop
                break
            else :
                # Clause improves hypothesis => continue
                H = new_H
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
        best_score = None
        try :
            # While there are untested refinements in the beam.
            while beam.has_active() :
              with Log('iteration', _timer=True) :
          
                # Create the next beam
                new_beam = beam.create()
        
                # Run through old beam and process its content
                for old_rule, refs in beam :
                
                  with Log('refining', rule=old_rule, score=old_rule.score, localScore=old_rule.localScore, _timer=True) :
                    if best_score != None and old_rule.localScoreMax < best_score :
                        # Prune this rule, it can never reach the top of the beam.
                        with Log('abort', reason="maxscore", bestscore=best_score, maxscore=old_rule.localScoreMax) : pass
                        continue 
            
                    # Add current rule to beam and mark it as tested (refinements=None)
                    new_beam.push( old_rule, None )
                        
                    # Update scores of available refinements and add new refinements if a new variable was introduced.
                    new_rules = self.update_refinements(old_rule, refs, best_score)
            
                    # Extract refinement literals
                    new_refs = [ r.literal for r in new_rules ]
                            
                    # Add rules to new beam (new_refs are ordered by score, descending)
                    for i, new_rule in enumerate(new_rules) :
                        
                        # Update best score
                        current_score = new_rule.localScore
                        if best_score == None or  current_score > best_score :
                            best_score = current_score
                
                        if self.VERBOSE > 2 : print (new_rule, new_rule.score, new_rule.localScore)
                
                        # Early stopping
                        if new_rule.score.FP == 0 and new_rule.score.FN == 0 :
                           return new_rule   # we found a rule with maximal score => no better rule can be found
                        else :
                            if new_rule.score.FP == 0 :
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
        return beam.content[0][0]

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
                
            if new_score.TP <= prev_score.TP :
                # Rule doesn't cover any true positive examples => it's useless
                with Log('rejected', reason="TP", literal=r.literal, score=r.score, localScore=r.localScore ) : pass
            elif r.max_significance < self.SIGNIFICANCE :
                # Rule cannot reach required significance => it's useless
                with Log('rejected', reason="s", literal=r.literal, score=r.score, max_significance=r.max_significance ) : pass
            elif not r._new_vars and r.score_predict == r.parent.score_predict :
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


class PF1Score(object) :

    def __init__(self, correct, predict, m) :
        self.M_ESTIMATE_M = m
        self.max_x = 1
        self.TP = 0.0
        self.FP = 0.0
        self.FN = 0.0
        self.TN = 0.0
        self.P = 0.0
        self.N = 0.0
            
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
        self.localScore = self.m_estimate()
        self.localScoreMax = self.m_estimate_max()
        
    def m_estimate(self) :
        m = self.M_ESTIMATE_M
        return (self.TP + m * (self.P / (self.N + self.P))) / (self.TP + self.FP + m) 
        
    def m_estimate_max(self) :
        m = self.M_ESTIMATE_M
        return (self.TP + m * (self.P / (self.N + self.P))) / (self.TP + m) 
    
    def accuracy(self) :
        M = self.P + self.N
        return (self.TP + self.TN ) / M
        
    def __str__(self) :
        return '%.3g %.3g %.3g %.3g' % (self.TP, self.TN, self.FP, self.FN )
    
class ProbFOIL1(LearningProblem) :
    
    def __init__(self, *args, m_estimate_m=10, **kwdargs) :
        super().__init__(*args, **kwdargs)
        self.M_ESTIMATE_M = m_estimate_m
    
    def calculateScore(self, rule) :
        return PF1Score(rule.score_correct, rule.score_predict, self.M_ESTIMATE_M)

class ProbFOIL2(LearningProblem) :
    
    def __init__(self, *args, m_estimate_m=10, **kwdargs) :
        super().__init__(*args, **kwdargs)
        self.M_ESTIMATE_M = m_estimate_m
    
    def calculateScore(self, rule) :
        if not rule.previous :
            return PF1Score(rule.score_correct, rule.score_predict, self.M_ESTIMATE_M)
        else :
            # TODO store this information somewhere
            previous_prediction = rule.previous.score_predict
            if rule.previous.probability != 1 :
                p = rule.previous.probability
                previous_prediction = [ a + (b-a)*p   for a,b in zip(rule.previous.previous.score_predict, previous_prediction) ]
            return PF2Score(rule.score_correct, rule.score_predict, previous_prediction, self.M_ESTIMATE_M)

class PF2Score_Incremental(object):

    def _calc_y(self, p,l,u) :
        if l == u :
            # inactive
            return None
        else :
            v = (p-l) / (u-l)
            if v < 0 :
                return 0    # overestimate
            elif v > 1 :
                return 1    # underestimate
            else :
                return v    # correctable
    
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
        #               TP_x = x \sum_{i=n+1}^{M} dS_i = \sum_{i=1}^{n} pl_i 
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
        
        TP_previous = 0.0
        FP_previous = 0.0
        
        TP_base = 0.0
        FP_base = 0.0
        values = []
        
        dS_total = 0.0
        for p,l,u in zip(correct, predict_prev, predict ) :
            TP_previous += min(l,p)
            FP_previous += max(0,l-p)
            
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
                    FP_x = x * dS_running - pl_running
                    
                    score_x = self._m_estimate_m(TP_x, FP_x)
                
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
                FP_x = x * dS_running - pl_running
            
                TN_x = N - FP_x
                FN_x = P - TP_x
                
                    # TN_x = M - tau_p + sigma_p - (sigma_u - sigma_l) * x - sigma_l
                    # FN_x = tau_p - sigma_p - (tau_u - tau_l - sigma_u + sigma_l) * x - tau_l + sigma_l
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
        
    def accuracy(self) :
        M = self.P + self.N
        return (self.TP + self.TN ) / M
            
    def m_estimate(self) :
        return self._m_estimate_m(self.TP, self.FP)

    def m_estimate_max(self) :
        return self._m_estimate_m(self.TP, 0)

            
    def _m_estimate_m(self, TP, FP) :
        return (TP + self.mPNP) / (TP + FP + self.M_ESTIMATE_M) 

    def __str__(self) :
        return '%.3g %.3g %.3g %.3g' % (self.TP, self.TN, self.FP, self.FN )
        

class PF2Score_NonIncremental(object) :

    def _calc_y(self, p,l,u) :
        if l == u :
            return 0
        else :
            v = (p-l) / (u-l)
            if v < 0 :
                return 0
            elif v > 1 :
                return 1
            else :
                return v
                
    def __init__(self, correct, predict, predict_prev, m) :
        self.M_ESTIMATE_M = m
        self.MIN_RULE_PROB = 0.01
    
        values = sorted( (self._calc_y(p,l,u), p,l,u) for p,l,u in zip(correct, predict_prev, predict) )
      # with Log('calcscore') :
      #   with Log('values', _child=values) : pass
    
        P = 0.0
        N = 0.0
        ys = set([])
        for y, p, l, u in values :
            ys.add(y)
            P += p
            N += (1-p)
        
        # TODO incremental computation
        def score(x) :
            r = [ ((u-l)*x + l, p) for y,p,l,u in values ]
            TP = sum( ri for ri, pi in r if ri < pi ) + sum( pi for ri, pi in r if ri >= pi )
            #with Log('fp', lst=r, x=x) : pass
            FP = sum( ri - pi for ri, pi in r if ri > pi )
            TN = sum( 1 - pi for ri, pi in r if ri <= pi ) + sum ( 1-ri for ri, pi in r if ri > pi )
            FN = sum( pi - ri for ri, pi in r if ri <= pi )
            return TP, FP, TN, FN
    
        max_s = None
        max_x = None
    
        # for x1 in range(0,101) :
        #     x = float(x1) / 100.0
        for x in sorted(ys) :
            TP, FP, TN, FN = score(x)
            s = self._m_estimate(m, TP, TN, FP, FN, P, N)
            # with Log('candidate', x=x, score=s, TP=TP, FP=FP, TN=TN, FN=FN) : pass
            if x >= self.MIN_RULE_PROB and ( max_s == None or s > max_s ) :
                max_s = s
                max_x = x
        if max_x == None :
            max_x = 1
            TP, FP, TN, FN = score(max_x)
            max_s = self._m_estimate(m, TP, TN, FP, FN, P, N)
            
        if max_x > 1 - 1e-5 :
            max_x = 1
            
        self.max_s = max_s
        self.max_x = max_x
        self.TP, self.FP, self.TN, self.FN = score(max_x)
        self.P = P
        self.N = N
    
        self.maxTP = score(1.0)[0]
        self.localScore = self.m_estimate()
        self.localScoreMax = self.m_estimate_max()
        
    
        # with Log('best', x=max_x, score=max_s, TP=self.TP, FP=self.FP, TN=self.TN, FN=self.FN, m_est=self.m_estimate(m)) : pass         

    # def __eq__(self, other) :
    #     if other == None : return False
    #     return (self.TP, self.FP, self.TN, self.FN) == (other.TP, other.FP, other.TN, other.FN)
    
    def m_estimate(self) :
        m = self.M_ESTIMATE_M
        return (self.TP + m * (self.P / (self.N + self.P))) / (self.TP + self.FP + m) 
        
    def m_estimate_max(self) :
        m = self.M_ESTIMATE_M
        return (self.TP + m * (self.P / (self.N + self.P))) / (self.TP + m) 

    def accuracy(self) :
        M = self.P + self.N
        return (self.TP + self.TN ) / M
    

    def _m_estimate(self, m, TP, TN, FP, FN, P, N) :
        #if (TP == 0 and FP == 0 and m == 0) : m = 1
        return (TP + m * (P / (N + P))) / (TP + FP + m) 

    def __str__(self) :
        return '%.3g %.3g %.3g %.3g' % (self.TP, self.TN, self.FP, self.FN )

PF2Score = PF2Score_Incremental
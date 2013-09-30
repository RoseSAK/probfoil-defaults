#! /usr/bin/env python3

from rule import *

##############################################################################
###                      PARSE COMMAND LINE ARGUMENTS                      ###
##############################################################################

from argparse import ArgumentParser

import math
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

if __name__ == '__main__' :
    parser = ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--beamsize', type=int, default=5, dest='BEAM_SIZE', help='size of search beam')
    parser.add_argument('-m', type=int, dest='M_ESTIMATE_M',  default=10, help='value of m for m-estimate')
    parser.add_argument('--distinct_vars', action='store_true', dest='DISTINCT_VARS', help='enable distinct variables (EXPERIMENTAL)')
    parser.add_argument('--pvalue', type=float, dest='SIGNIFICANCE_P', default=0.99, help='minimal rule significance (p-value)')
    parser.add_argument('-v', action='count', dest='VERBOSE', help='increase verbosity level')
    parser.add_argument('--probfoil1', action='store_false', dest='PROBFOIL2', help='use traditional ProbFOIL scoring')
    parser.add_argument('--min_rule_prob', type=float, dest='MIN_RULE_PROB', default=0.01, help='minimal probability of rule in Prob2FOIL')
    parser.add_argument('--equiv_check', action='store_true', dest='EQUIV_CHECK', help='enable elimination of equivalent results in beam')

    SETTINGS = parser.parse_args()
    SETTINGS.SIGNIFICANCE= calc_significance(SETTINGS.SIGNIFICANCE_P)
    
    # TODO remove this hacky part
    import rule
    rule.SETTINGS = SETTINGS

##############################################################################
###                           LEARNING ALGORITHM                           ###
##############################################################################

class FOIL(object) :
    
    def __init__(self, SETTINGS) :
        self.SETTINGS = SETTINGS

    def learn(self, H) :
        """Core FOIL learning algorithm.
    
        H - initial hypothesis (Rule (e.g. FalseRule))
        """
    
        # Find clauses as long as stopping criterion is not met or until maximal score (1.0) is reached.
        while H.globalScore < 1.0 :   # this test is not required for correctness (alternative: while True)
       
          with Log('learn_rule', _timer=True):
            
            # Find best clause that refines this hypothesis
            new_H = self.best_clause( H )
        
            if self.SETTINGS.VERBOSE : print ('RULE FOUND:', new_H, new_H.globalScore)
        
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
        beam = Beam(self.SETTINGS.BEAM_SIZE)    
    
        # Create the new rule ( target <- )
        init_rule = RuleHead(current_rule.target, previous=current_rule, learner=self)
    
        # Calculate initial set of refinements
        refinements = list(init_rule.refine())
    
        # Add clause to beam (with empty score)
        beam.push( init_rule, refinements )
    
        # While there are untested refinements in the beam.
        while beam.has_active() :
          with Log('iteration', _timer=True) :
          
            # Create the next beam
            new_beam = beam.create()
        
            # Run through old beam and process its content
            for old_rule, refs in beam :
            
              with Log('refining', rule=old_rule, score=old_rule.score, localScore=old_rule.localScore, _timer=True) :
            
                # Add current rule to beam and mark it as tested (refinements=None)
                new_beam.push( old_rule, None )
                        
                # Update scores of available refinements and add new refinements if a new variable was introduced.
                new_rules = self.update_refinements(old_rule, refs)
            
                # Extract refinement literals
                new_refs = [ r.literal for r in new_rules ]
            
                # Add rules to new beam (new_refs are ordered by score, descending)
                for i, new_rule in enumerate(new_rules) :
                
                    if self.SETTINGS.VERBOSE : print (new_rule, new_rule.score, new_rule.localScore)
                
                    if new_rule.score.FP == 0 and new_rule.score.FN == 0 :
                       return new_rule   # we found a rule with maximal score => no better rule can be found
                    # elif new_rule.localScore <= old_rule.localScore) :
                    #     break
                    # Attempt to add rule to beam
                    elif not new_beam.push( new_rule , new_refs[i+1:] ) : 
                        break  # current ref does not have high enough score -> next ones will also fail
            
            # Use new beam in next iteration
            beam = new_beam
        
            # Write beam to log
            with Log('beam', _child=beam.toXML()) : pass
    
        # Return head of beam.
        return beam.content[0][0]

    def update_refinements(self, rule, refine) :
        #with Log('ref', rule=rule, refine=refine) : pass
        if refine == None :
            return []
    
        if self.SETTINGS.DISTINCT_VARS :
            new_refine = list(rule.refine(update=False))
            refine = new_refine
        else :
            # Calculate new refinements in case a variable was added by the previous literal
            new_refine = list(rule.refine(update=True))
            # Add new refinements
            refine += new_refine
        
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
            elif r.max_significance < self.SETTINGS.SIGNIFICANCE :
                # Rule cannot reach required significance => it's useless
                with Log('rejected', reason="s", literal=r.literal, score=r.score, max_significance=r.max_significance ) : pass
            else :
                # Accept the extension and add it to the output
                with Log('accepted', literal=r.literal, score=r.score, localScore=r.localScore ) : pass
                result.append( r )
    
        return result
    
    def calculateScore( self, rule ) :    
        predict = [ x[0] for x in rule.evaluate() ]
        correct = [ x[0] for x in rule.examples]
        return FOIL.Score(correct, predict)
    
    # ProbFOIL/mFOIL score calculation
    class Score(object) :
    
        def __init__(self, correct, predict) :
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
            
        def m_estimate(self, m) :
            return (self.TP + m * (self.P / (self.N + self.P))) / (self.TP + self.FP + m) 
        
        def accuracy(self) :
            M = self.P + self.N
            return (self.TP + self.TN ) / M
            
        def __str__(self) :
            return '%.3g %.3g %.3g %.3g' % (self.TP, self.TN, self.FP, self.FN )
     
class Prob2FOIL(FOIL) :
    
    # Prob2FOIL score calculation   
    class Score(FOIL.Score) :
    
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
    
    
        def __init__(self, correct, predict, predict_prev) :
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
            m = SETTINGS.M_ESTIMATE_M

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
                if x >= SETTINGS.MIN_RULE_PROB and ( max_s == None or s > max_s ) :
                    max_s = s
                    max_x = x
            if max_x == None :
                max_x = 1
                TP, FP, TN, FN = score(max_x)
                max_s = self._m_estimate(m, TP, TN, FP, FN, P, N)

            self.max_s = max_s
            self.max_x = max_x
            self.TP, self.FP, self.TN, self.FN = score(max_x)
            self.P = P
            self.N = N
        
            self.maxTP = score(1.0)[0]
        
            # with Log('best', x=max_x, score=max_s, TP=self.TP, FP=self.FP, TN=self.TN, FN=self.FN, m_est=self.m_estimate(m)) : pass         
        
    
        def _m_estimate(self, m, TP, TN, FP, FN, P, N) :
            #if (TP == 0 and FP == 0 and m == 0) : m = 1
            return (TP + m * (P / (N + P))) / (TP + FP + m) 
    
    
    def __init__(self, SETTINGS) :
        super(Prob2FOIL,self).__init__(SETTINGS)
        
    def calculateScore(self, rule ) :
        predict = [ x[0] for x in rule.evaluate() ]
        correct = [ x[0] for x in rule.examples]
    
        if rule.previous :
            predict_prev = [ x[0] * rule.previous.probability for x in rule.previous.evaluate() ]
        else :
            predict_prev = [ 0 for x in rule.examples ]
        return Prob2FOIL.Score(correct, predict, predict_prev)           

##############################################################################
###                            SCRIPT EXECUTION                            ###
##############################################################################

def read_file(filename) :
    if filename.endswith('.arff') :
        return read_arff(filename)
    else :
        return read_prolog(filename)

def read_prolog(filename) :
    import re 
    
    line_regex = re.compile( "((?P<type>(base|modes|learn))\()?((?P<prob>\d+[.]\d+)::)?\s*(?P<name>\w+)\((?P<args>[^\)]+)\)\)?." )
    
    kb = KnowledgeBase([])
    
    with open(filename) as f :
        for line in f :        
            if line.strip().startswith('%') :
                continue
            m = line_regex.match(line.strip())
            if m : 
                ltype, pred, args = m.group('type'), m.group('name'), list(map(lambda s : s.strip(), m.group('args').split(',')))
                prob = m.group('prob')
                
                if ltype == 'base' :
                    kb.register_predicate(pred, args)  
                elif ltype == 'modes' :
                    kb.add_mode(pred,args)                    
                elif ltype == 'learn' :
                    kb.add_learn(pred,args)                
                else :
                    if not prob :
                        prob = 1.0
                    else :
                        prob = float(prob)
                    kb.add_fact(pred,args,prob)
                                
    return kb
    
def read_arff(filename) :
    kb = KnowledgeBase([])
    
    atts = []
    with open(filename) as f :
        reading_data = False
        i = 0
        for line in f :
            line = line.strip()
            if line.startswith('#') :
                continue    # comment
            elif line.startswith('@attribute') :
                _a, name, tp = line.split()
                assert(tp == 'numeric')
                atts.append( name )
                kb.register_predicate(name, ['ex'])
            elif line.startswith('@data') :
                reading_data = True
            elif line and reading_data :
                data = map(float,line.split(','))
                
                for a,v in zip(atts,data) :
                    kb.add_fact(a,[str(i)],v)
                i += 1

        for a in atts[:-1] :
            kb.add_mode(a, ['+'])
        kb.add_learn(atts[-1], ['ex'])
    return kb

def main(files=[]) :

    LSETTINGS = dict(vars(SETTINGS))
    del LSETTINGS['files']

    
    for filename in files :
        
        kb = read_file(filename)
        
        varnames = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        targets = kb.learn
        
        filename = os.path.split(filename)[1]
        
        if SETTINGS.PROBFOIL2 :
            lp = Prob2FOIL(SETTINGS)
        else :
            lp = FOIL(SETTINGS)
        
        with open(filename+'.xml', 'w') as Log.LOG_FILE :
            
            with Log('log') :
                for pred, args in targets :
                  with Timer('learning time') as t :
                    
                    kb.idtypes = args
                    kb.reset_examples()
                    
                    target = Literal(pred, varnames[:len(args)] )
                    
                    print('==> LEARNING CONCEPT:', target)
                    with Log('learn', input=filename, target=target, _timer=True, **LSETTINGS ) :
                        H = lp.learn(FalseRule(kb, target,lp))   
                        
                        print (H.strAll())
                        print(H.score, H.globalScore)
                        
                        with Log('result') as l :
                            if l.file :
                                print(H.strAll(), file=l.file)
                                

if __name__ == '__main__' :
    main(SETTINGS.files)

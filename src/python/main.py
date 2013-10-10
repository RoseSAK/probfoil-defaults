#! /usr/bin/env python3

import sys
from util import Log

from language import Literal, Language, RootRule
from prolog_interface import PrologInterface
from learn import ProbFOIL2, ProbFOIL

def test(filename, target, *modes) :
    
    # from rule import Literal, RuleHead, FalseRule
    
    target_pred, target_arity = target.split('/')
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    target_args = [] 
    for x in range(0, int(target_arity)) :
        target_args.append(letters[x])
    target = Literal(target_pred, target_args)
    
    modes = list(map(lambda x : Literal(*x.split('/')), modes))
      
    with open('log.xml', 'w') as Log.LOG_FILE : 
     with Log('log') :
        import prolog.parser as pp
        parser = pp.PrologParser()
    
        p = PrologInterface()
    
        p.engine.loadFile(filename)
    
        l = Language()
        
        l.initialize(p)  # ==> read language specification + values from p
                
        for mode in modes :
            l.setArgumentModes( mode )
    
        lp = ProbFOIL2(l, p)
        
        with Log('initialize', _timer=True) :
            r0 = RootRule(target, lp)
            r0.initialize()
        
        print(r0.refine())
        
        print (r0.score_correct)
        
        try :
            result = lp.learn(r0)
        except :
            with Log('grounding_stats', **vars(p.engine.ground_cache.stats())) : pass
            with Log('error') : pass
#            p.engine.listing()
            with open('/tmp/probfoil.pl','w') as pl_out :
                print (p.engine.listing(), file=pl_out)

            raise Exception('ERROR')

        with Log('grounding_stats', **vars(p.engine.ground_cache.stats())) : pass

        print('######################################################')
        print('###################     RESULT     ###################')
        print('######################################################')
        print(str(result).replace('\t','\n'))
        
        with open('/tmp/probfoil.pl','w') as pl_out :
            print (p.engine.listing(), file=pl_out)

        #print(p.engine.ground_cache)

if __name__ == '__main__' :
    test(*sys.argv[1:])    
#! /usr/bin/env python3

import sys, time
from util import Log

from language import Literal, Language, RootRule
from prolog_interface_yap import YapPrologInterface
from learn import ProbFOIL2, ProbFOIL

def main(filename, target, *modes) :
    
    # from rule import Literal, RuleHead, FalseRule
    
    target_pred, target_arity = target.split('/')
    target_arity = int(target_arity)
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    target_args = [] 
    for x in range(0, target_arity) :
        target_args.append(letters[x])
    target = Literal(target_pred, target_args)
    
    modes = list(map(lambda x : Literal(*x.split('/')), modes))
      
    with open('log.xml', 'w') as Log.LOG_FILE : 
     with Log('log') :
        # import prolog.parser as pp
        # parser = pp.PrologParser()
    
        p = YapPrologInterface()
    
        p.engine.loadFile(filename)
    
        l = Language()
                
        for mode in modes :
            l.setArgumentModes( mode )
            
        l.addTarget( target_pred, target_arity )
        
        l.initialize(p)  # ==> read language specification + values from p
        
        lp = ProbFOIL2(l, p)
        
        with Log('initialize', _timer=True) :
            r0 = RootRule(target, lp)
            r0.initialize()
        
        print(r0.refine())
        
        print (r0.score_correct)
        
        learn_time = time.time()
        try :
            result = lp.learn(r0)
        except :
            with Log('grounding_stats', **vars(p.engine.getGrounding().stats())) : pass
            with Log('error') : pass
#            p.engine.listing()
            # with open('/tmp/probfoil.pl','w') as pl_out :
            #     print (p.engine.listing(), file=pl_out)

            raise Exception('ERROR')

        with Log('grounding_stats', **vars(p.engine.getGrounding().stats())) : pass

        print('##################################################################')
        print('#########################     RESULT     #########################')
        print('##################################################################')
        print(str(result).replace('\t','\n'))
        
        # with open('/tmp/probfoil.pl','w') as pl_out :
        #     print (p.engine.listing(), file=pl_out)

        learn_time = time.time() - learn_time
        for t in Log.TIMERS :
            print( '%s => %.3fs (%.3f%%)' % (t, Log.TIMERS[t], 100*(Log.TIMERS[t] / learn_time) ))    
        print('total',' => ', learn_time, 's', sep='')

if __name__ == '__main__' :
    main(*sys.argv[1:])    
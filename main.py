#! /usr/bin/env python3


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


import sys, time
from util import Log, WorkEnv, Timer
import os

from language import Literal, Language, RootRule
from prolog_interface_yap import YapPrologInterface
from learn import ProbFOIL2, ProbFOIL1

def parse_args(args) :
    
    import argparse
    
    p = argparse.ArgumentParser(description="ProbFOIL learning algorithm")
    p.add_argument('input', metavar='FILE', help="Input data file.")
    p.add_argument('target', metavar='TARGET', help="Target to learn as pred/arity (e.g. grandmother/2).")
    p.add_argument('modes', metavar='MODE', nargs='+', help="Extension modes as pred/argmodes (e.g. mother/+-).")
    p.add_argument('-m','--m_estimate_m', type=int, default=10, help="Value for m in m-estimate calculation.")
    p.add_argument('-b','--beam_size', type=int, default=5, help="Size of search beam.")
    p.add_argument('-p','--significance_p_value', type=float, default=0.99, help="P-value for rule significance.")
    p.add_argument('-v','--verbose', action='count', help="Verbosity level.", default=0)
    p.add_argument('-s','--probfoil', choices=['1','2'], default='2', help="Scoring function for ProbFOIL version (1/2)")
    
    return p.parse_args(args)

def main(arguments) :
    
    args = parse_args(arguments)
    
    if 'PROBLOGPATH' in os.environ :
        PROBLOGPATH = os.environ['PROBLOGPATH']
    else :
        try :
            import settings
            PROBLOGPATH = settings.PROBLOGPATH
        except Exception :
            print('PROBLOGPATH environment variable not set. Set it with \'export PROBLOGPATH=<path to problog>\' or define it in the file \'settings.py\'.', file=sys.stderr)
            sys.exit(1)
    
    parameters = vars(args)
        
    target_pred, target_arity = args.target.split('/')
    target_arity = int(target_arity)
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    target_args = [] 
    for x in range(0, target_arity) :
        target_args.append(letters[x])
    target = Literal(target_pred, target_args)
    
    modes = list(map(lambda x : Literal(*x.split('/')), args.modes))
      
    with open('log.xml', 'w') as Log.LOG_FILE :
     with WorkEnv(PROBLOGPATH=PROBLOGPATH) as env :    # Set up a temporary working directory
      with Log('log', **parameters) :
        p = YapPrologInterface(env)
        
        p.engine.loadFile(args.input)
        
        init_time1 = time.time()
        
        with Timer(category='init_language') :
            l = Language()
            for mode in modes :
                l.setArgumentModes( mode )
            l.addTarget( target_pred, target_arity )
            l.initialize(p)  # ==> read language specification + values from p        

        with Timer(category='init_learner') :
            if args.probfoil == '2' :
                lp = ProbFOIL2(l, p, **parameters)
            else :
                lp = ProbFOIL1(l, p, **parameters)
        
            if args.verbose : print('Initializing root rule...')
            with Log('initialize', _timer=True) :
                r0 = RootRule(target, lp)
                r0.initialize()
            
        if args.verbose > 1 :
            print (r0.score_correct)
        
        if args.verbose: print('Start learning...')
        learn_time = time.time()
        try :
            result = lp.learn(r0)
        except Exception as e :
            with Log('grounding_stats', **vars(p.engine.getGrounding().stats())) : pass
            with Log('error') : pass
            raise e

        with Log('grounding_stats', **vars(p.engine.getGrounding().stats())) : pass

        print('##################################################################')
        print('#########################     RESULT     #########################')
        print('##################################################################')
        if result.getTheory() :
            print('\n'.join(result.getTheory()))
        else :
            print('%s :- fail.' % result.target )
        print('#########################     SCORES     #########################')
        print('PREDICTIONS (TP, TN, FP, FN) :', result.score)
        print('ACCURACY                     :', result.globalScore)
        
        print('#########################     TIMING     #########################')
        learn_time = time.time() - learn_time
        for t in Timer.TIMERS :
            print( '%s => %.3fs (%.3f%%)' % (t, Timer.TIMERS[t], 100*(Timer.TIMERS[t] / learn_time) ))    
        print('total',' => ', learn_time, 's', sep='')

if __name__ == '__main__' :
    main(sys.argv[1:])    
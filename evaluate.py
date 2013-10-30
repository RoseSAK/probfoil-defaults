#! /usr/bin/env python3
import sys, os


def main( probfoil_output_file, test_file ) :

    if 'PROBLOGPATH' in os.environ :
        PROBLOGPATH = os.environ['PROBLOGPATH']
    else :
        try :
            import settings
            PROBLOGPATH = settings.PROBLOGPATH
        except Exception :
            print('PROBLOGPATH environment variable not set. Set it with \'export PROBLOGPATH=<path to problog>\' or define it in the file \'settings.py\'.', file=sys.stderr)
            sys.exit(1)
    
    sys.path.append(PROBLOGPATH + '/src/')
    
    from problog import ProbLogEngine
    from utils import Timer, Logger, WorkEnv
    
    engine = ProbLogEngine.create([])
    
    with WorkEnv(None,Logger(), persistent=WorkEnv.NEVER_KEEP) as env :
        
        file_in = env.tmp_path('model.pl')
        
        with open(file_in, 'w') as f_out :
            with open(test_file) as f_in :
                print( f_in.read(), file=f_out )
            with open(probfoil_output_file) as f_in :
                print( f_in.read(), file=f_out )
        
        result = engine.execute(file_in, env)
        
        evaluate( result.items() )

def evaluate( problog_results ) :
            
    prefix = 'pf_eval_'
        
    correct = {}
    predict = {}
        
    for key, value in problog_results :
        if key.startswith(prefix) :
            key = key[len(prefix):]
            predict[key] = float(value)
        else :
            correct[key] = float(value)
    
    assert( len(correct) == len(predict) )
    
    pairs = [ (correct[k], predict[k]) for k in correct ]
    
    P = 0
    M = 0
    TP = 0
    FP = 0
    
    for c, p in pairs :
        P += c
        M += 1        
        TP += min(c,p)   
        FP += max(p-c,0)

    N = M - P
    TN = N-FP
    FN = P-TP
    
    accuracy = (TP + TN) / M
    
    print (accuracy, '|', TP, TN, FP, FN)

if __name__ == '__main__' :
    main(*sys.argv[1:3])
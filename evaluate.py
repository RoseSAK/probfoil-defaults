#! /usr/bin/env python3
import sys, os

def arff_to_pl(filename_in, file_out) :
    with open(filename_in) as file_in :
        line_num = 0
        for line_in in file_in :
            line_in = line_in.strip()
            if line_in and not line_in.startswith('@') and not line_in.startswith('#') :
                values = list(map(float,line_in.split(',')))
                num_atts = len(values)
                line_out = '\n'.join( '%.6f::att%s(%s).' % (float(val), att, line_num) for att, val in enumerate(values) ) + '\n\n'
                
                if line_num == 0 :
                    # write LEARN header
                    line_out = '%%LEARN att%s/1 ' % (len(values)-1) + ' '.join( 'att%s/+' % att for att, val in enumerate(values[:-1]) )  + '\n'
                    file_out.write(line_out)
                line_out = '\n'.join( '%.6f::att%s(%s).' % (float(val), att, line_num) for att, val in enumerate(values) ) + '\n\n'
                file_out.write(line_out)
                line_num += 1
        line_out = '\n'.join( 'base(att%s(id)).' % att for att in range(0, num_atts) ) + '\n\n'
        file_out.write(line_out)    


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
    
    engine = ProbLogEngine.create(['-d', 'c2d'])
    
    with WorkEnv(None,Logger(), persistent=WorkEnv.NEVER_KEEP) as env :
        
        # First evaluate 'correct'

        file_in = env.tmp_path('model.pl')

        with open(probfoil_output_file) as f_in :
            for line in f_in :
                line = line.strip()
                if line.startswith('%TARGET: ') :
                    target = line[9:]
                    break

        
        with open(file_in, 'w') as f_out :
            if test_file.endswith('.arff') :
                arff_to_pl(test_file, f_out)
            else :
                with open(test_file) as f_in :
                    print( f_in.read(), file=f_out )
            print ('query(%s).' % target, file=f_out )
                    
        result_correct = engine.execute(file_in, env)
        
        with open(file_in, 'w') as f_out :
            if test_file.endswith('.arff') :
                arff_to_pl(test_file, f_out)
            else :
                with open(test_file) as f_in :
                    print( f_in.read(), file=f_out )
            with open(probfoil_output_file) as f_in :
                print (f_in.read(), file=f_out)
            
            for x in result_correct :
                print ('query(pf_eval_%s).' % x, file=f_out )
            
        result_predict = engine.execute(file_in, env)
        
        result_all = {}
        result_all.update(result_correct)
        result_all.update(result_predict)
        
        
        evaluate( result_all.items() )

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
    
    if len(correct) != len(predict) :
        print (correct, predict)
        raise Exception('Number of predictions does not match number of examples.')
        
    
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
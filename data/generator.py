#! /usr/bin/env python3

import sys
# Dataset generator for ProbFOIL

# Parameters:
#   target arity        : 1
#   number of values
#   predicate arities and density
#   percentage noise
#   sparsety
#   number of rules

# p::att1(1).
# p::att1(2).
# p::att1(3).
# ...
# 
# 
# p1::t(X) <- att1(X), att2(X), att3(X).
# p2::t(X) <- \+att2(X), att(5(X).
# 
# => evaluate this and add noise




def generateFacts(predicate, arity, values, density) :

    from itertools import product
    from random import random
    
    for args in product(*[values]*arity) : 
        if random() <= density :
            yield '%.4f::att%s(%s).' % ( random(), predicate, ','.join(map(str,args)) )
            
def main( num_values ) :
    
    num_values = int(num_values)
    
    target_arity = 2
    fact_arities = [2,2,1,1,2,1,2]
    density = 1
    rule_lengths = [2,2,3]
    
    facts = []
    
    print ( 'base(t(%s)).' % (','.join(['id'] * target_arity) ) )
    for att_id, att_arity in enumerate(fact_arities) :
        print ( 'base(att%s(%s)).' % (att_id, ','.join(['id'] * att_arity) ) )
    
    
    for att_id, att_arity in enumerate(fact_arities) :
        for f in  generateFacts( att_id, att_arity, range(0,num_values), density**att_arity) :
            print (f)

    print( '0.85::t(A,B) <- att0(A,C), att3(C), att4(C,B).')
    print( '0.50::t(A,B) <- \+att2(B), att6(B,A).')


    
def main2( num_values ) :
    
    num_values = int(num_values)
    
    target_arity = 1
    fact_arities = [1,1,1,1,1,1,1]
    density = 1
    # rule_lengths = [2,2,3]
    
    facts = []
    
    print ( 'base(t(%s)).' % (','.join(['id'] * target_arity) ) )
    for att_id, att_arity in enumerate(fact_arities) :
        print ( 'base(att%s(%s)).' % (att_id, ','.join(['id'] * att_arity) ) )
    
    
    for att_id, att_arity in enumerate(fact_arities) :
        for f in  generateFacts( att_id, att_arity, range(0,num_values), density**att_arity) :
            print (f)

    print( '0.85::t(A) <- att0(A), att3(A), att4(A).')
    print( '0.50::t(A) <- \+att2(A), att6(A).')
#    print('\n'.join(facts))
    
    
    
if __name__ == '__main__' :
    main2(*sys.argv[1:])        

#! /usr/bin/env python3

def localStop(H) :
    
    TP_p = H.TP1
    TP_c = H.TP
    
    FP_p = H.FP1
    FP_c = H.FP
    
    #print H.POS, H.NEG
    
    return ( TP_c - TP_p == 0 ) or ( FP_c - FP_p == 0 )

def localScore(rs) :
    m = 1
    
    m_estimate_c = (rs.TP + m * (rs.P / (rs.N + rs.P))) / (rs.TP + rs.FP + m) 
    
    m_estimate_p = (rs.TP1 + m * (rs.P / (rs.N + rs.P))) / (rs.TP1 + rs.FP1 + m) 
    
    return m_estimate_c - m_estimate_p
    
def globalScore(rs) :
    return (rs.TP + rs.TN ) / (rs.TP + rs.TN + rs.FP + rs.FN)
    
def learn(H) :

    score_H = globalScore(H)

    while True :
        H.pushRule()
        while not localStop(H) :
            lit = best_literal( H, H.rules[-1].refine(H.data), lambda H : localScore(H) ) 
            #if lit == None : return H
            H.pushLiteral(lit)

        
        # FIXME pruning step
        # n = len(r.body)
        # 
        # i = argmax( range(1,n+1), lambda i : localScore(H, H.newRule(target,r.body[:i]), data ))
        # 
        # c = H.newRule( r.head, r.body[:i] )        
        # newH = H + c
        
        score_nH = globalScore(H)
        
        #print score_H, score_nH
        if (score_H >= score_nH) :
            H.popRule()
            break
        else :
            score_H = score_nH
    return H

def best_literal( H, generator, score_func ) :
    max_score = None
    max_x = None
    
    for lit in generator :
        H.pushLiteral(lit)
        current_score = score_func(H)
        if max_score == None or max_score < current_score :
            max_score, max_x = current_score, lit
        H.popLiteral()
#    lit = argmax(r.refine(data), lambda lt : localScore(H, r+lt, data)) 
    return max_x



        
def argmax( generator, score ) :
    max_score = None
    max_x = None
    
    refs = list(generator)
    
#    print 'ARGMAX'
    for x in refs :
        current_score = score(x)
#        print 'REF', x, current_score
        if max_score == None or max_score < current_score :
            max_score = current_score
            max_x = x
#    print '==>', max_x, max_score
    return max_x
        


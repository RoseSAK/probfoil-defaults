#! /usr/bin/env python3

from __future__ import print_function

import sys, time

class Log(object) :
    
    LOG_FILE=sys.stderr
    
    def __init__(self, tag, file=None, _child=None, _timer=False, **atts) :
        if file == None :
            file = Log.LOG_FILE
        self.tag = tag
        self.atts = atts
        self.file = file
        self._child = _child
        if _timer :
            self._timer = time.time()            
        else :
            self._timer = None

    
    def get_attr_str(self, atts=None) :
        string = ''
        for k in self.atts :
            v = self.atts[k]
            #if hasattr(v,'__call__') :
            #    v = v()
            string += '%s="%s" ' % (k, v)
        return string
                    
    def __enter__(self) :
        if self.file :
            print('<%s %s>' % (self.tag, self.get_attr_str()), file=self.file)
            if self._child != None : print(self._child, file=self.file)
        return self
        
    def __exit__(self, *args) :
        if self.file :
            if self._timer != None :
                print('<runtime time="%.5f"/>' % (time.time() - self._timer), file=self.file )
            print('</%s>' % (self.tag,), file=self.file)
            
class Timer(object) :
    
    def __init__(self, desc) :
        self.desc = desc
    
    elapsed_time = property(lambda s : time.time()-s.start)
    
    def __enter__(self) :
        self.start = time.time()
        return self
        
    def __exit__(self, *args) :
        print ( '%s: %.5fs' % (self.desc, self.elapsed_time ))


class Beam(object) :
    
    def __init__(self, size, allow_equivalent=False) :
        self.size = size
        self.content = []
        self.allow_equivalent = allow_equivalent
         
    def create(self) :
        return Beam(self.size, self.allow_equivalent) 
       
    def __iter__(self) :
        return iter(self.content)
        
    def push(self, obj, active) :
        if len(self.content) == self.size and obj < self.content[-1][0] : return False
        
        is_last = True
        
        p = len(self.content) - 1
        self.content.append( (obj, active) )
        while p >= 0 and (self.content[p][0] == None or self.content[p][0] < self.content[p+1][0]) :
            self.content[p], self.content[p+1] = self.content[p+1], self.content[p] # swap elements
            p = p - 1
            is_last = False
        
        if not self.allow_equivalent and len(self.content) > 1 :
            r1, rf1 = self.content[p]
            r2, rf2 = self.content[p+1]
            
            r1scores = r1.score_predict
            r2scores = r2.score_predict
            
            if r1.localScore == r2.localScore and r1scores == r2scores :
                if rf1 != None and rf2 != None and len(rf1) > len(rf2) : #len(r1.variables) > len(r2.variables) :                
                    best, worst = r1, r2
                    self.content[p+1] = self.content[p]
                else :
                    best, worst = r2, r1
                with Log('beam_equivalent', best=best, worst=worst) : pass                
                self.content.pop(p)
        
        popped_last = False
        while len(self.content) > self.size :
            self.content.pop(-1)
            popped_last = True
            
        return not (is_last and popped_last)
    
    def peak_active(self) :
        i = 0
        while i < len(self.content) :
            if self.content[i][-1] :
                yield self.content[i]
                i = 0
            else :
                i += 1
                
    def has_active(self) :
        for r, act in self :
            if act != None : return True
        return False
    
    def pop(self) :
        self.content = self.content[1:]
        
    def __str__(self) :
        res = ''
        for c, r in self.content :
            res += str(c) + ': ' + str(c.score) +  ' | ' + str(r) + '\n'
        return res
        
    def toXML(self) :
        res = ''
        for c, r in self.content :
            if r == None :
                res +=  '<record rule="%s" score="%s" localScore="%s" refine="NO" />\n' % (c,c.score, c.localScore)
            else :
                res +=  '<record rule="%s" score="%s" localScore="%s" refine="%s" />\n' % (c,c.score, c.localScore, '|'.join(map(str,r)))
        return res

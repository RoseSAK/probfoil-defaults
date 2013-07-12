#! /usr/bin/env python3

from __future__ import print_function

import sys, time

class Log(object) :
    
    LOG_FILE=sys.stderr
    
    def __init__(self, tag, file=None, **atts) :
        if file == None :
            file = Log.LOG_FILE
        self.tag = tag
        self.atts = atts
        self.file = file
    
    def get_attr_str(self, atts=None) :
        string = ''
        for k in self.atts :
            v = self.atts[k]
            #if hasattr(v,'__call__') :
            #    v = v()
            string += '%s="%s" ' % (k, v)
        return string
    
    def logline(self) :
        if self.file :
            print('<%s %s/>' % (self.tag, self.get_attr_str()), file=self.file)
            
    def logXML(self, xml) :
        with self :
            if self.file :
                print(xml, file=self.file)
    
    def __enter__(self) :
        if self.file :
            print('<%s %s>' % (self.tag, self.get_attr_str()), file=self.file)
        return self
        
    def __exit__(self, *args) :
        if self.file :
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
        
    def push(self, obj, active, score) :
        if len(self.content) == self.size and score < self.content[-1][0] : return False
        
        is_last = True
        
        p = len(self.content) - 1
        self.content.append( (score, obj, active) )
        while p >= 0 and (self.content[p][0] == None or self.content[p][0] < self.content[p+1][0]) :
            self.content[p], self.content[p+1] = self.content[p+1], self.content[p] # swap elements
            p = p - 1
            is_last = False
        
        if not self.allow_equivalent and p >= 0 and self.content[p][0] == self.content[p+1][0] :
            worst, best = sorted( [self.content[p], self.content[p+1]] )
            Log('beam_equivalent', best=best[1], worst=worst[1]).logline()
            self.content[p+1] = best
            self.content = self.content[:p] + self.content[p+1:]    # remove p    
        
        popped_last = False
        while len(self.content) > self.size :
            self.content.pop(-1)
            popped_last = True
            
        return not (is_last and popped_last)
    
    def peak_active(self) :
        i = 0
        while i < len(self.content) :
            if self.content[i][2] :
                yield self.content[i]
                i = 0
            else :
                i += 1
                
    def has_active(self) :
        for s, r, act in self :
            if act : return True
        return False
    
    def pop(self) :
        self.content = self.content[1:]
        
    def __str__(self) :
        res = ''
        for s, c, r in self.content :
            res += str(c) + ': ' + str(s) +  ' | ' + str(r) + '\n'
        return res
        
    def toXML(self) :
        res = ''
        for s, c, r in self.content :
            if r == None :
                res +=  '<record rule="%s" score="%s" refinements="" />\n' % (c,s)
            else :
                res +=  '<record rule="%s" score="%s" refinements="%s" />\n' % (c,s,'|'.join(map(lambda s : str(s[1]), r)))
        return res

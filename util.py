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

from __future__ import print_function

import sys, time, os
from collections import defaultdict
import tempfile, shutil

class WorkEnv(object) :
    
    NEVER_KEEP=0        # directory is always removed on exit
    KEEP_ON_ERROR=1     # directory is removed, unless exit was caused by an error
    ALWAYS_KEEP=2       # directory is never removed
    # NOTE: a pre-existing directory is never removed
    
    def __init__(self, outdir=None, persistent=KEEP_ON_ERROR, **extra) :
        self.__outdir = outdir
        self.__persistent = persistent
        self.__extra = extra
        
    logger = property(lambda s : s.__logger)
    
    def __getitem__(self, key) :
        return self.__extra[key]
        
    def __setitem__(self, key, value) :
        self.__extra[key] = value
        
    def __contains__(self, key) :
        return key in self.__extra 
    
    def __enter__(self) :
        if self.__outdir == None :
            self.__outdir = tempfile.mkdtemp()
        elif not os.path.exists(self.__outdir) :
            os.makedirs(self.__outdir)
        else :  # using non-temporary, existing directory => NEVER delete this
            self.__persistent = self.ALWAYS_KEEP
        return self
        
    def __exit__(self, exc_type, value, traceback) :
        if self.__persistent == self.NEVER_KEEP or (self.__persistent == self.KEEP_ON_ERROR and exc_type == None) :
            shutil.rmtree(self.__outdir)
        elif self.__persistent == self.KEEP_ON_ERROR and exc_type != None :
            # An error occurred
            print('Error occurred: working directory preserved', self.__outdir, file=sys.stderr)
        
    def out_path(self, relative_filename) :
        return os.path.join(self.__outdir, relative_filename)
        
    def tmp_path(self, relative_filename) :
        return os.path.join(self.__outdir, relative_filename)

class Log(object) :
    
    LOG_FILE=sys.stderr
    TIMERS=defaultdict(float)
    
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
                res +=  '<record rule="%s" score="%s" localScore="%s" maxScore="%s" refine="NO" />\n' % (c,c.score, c.localScore, c.localScoreMax)
            else :
                res +=  '<record rule="%s" score="%s" localScore="%s" maxScore="%s" refine="%s" />\n' % (c,c.score, c.localScore, c.localScoreMax, '|'.join(map(str,r)))
        return res

#!/usr/bin/env python3
# encoding: utf-8
"""
evaluatennf.py

Created by Wannes Meert on 06-12-2012.
Based on code by Guy Van den Broeck.
Copyright (c) 2012 KULeuven. All rights reserved.
"""

from __future__ import print_function

import sys
import getopt
import math
import re
import pprint

import math


try :
    import numpy as np
    np.seterr(all='ignore')
    numpy_available = True
except ImportError :
    numpy_available = False

verbose = False
def printv(string):
    if verbose: print(string)

class MathNP(object) :
        
    def __init__(self, dimensions) :
        self.dimensions = dimensions
        
    def logsumexp(self,loga, logb):
        x = float('-inf')
        e = np.where( np.logical_and( np.isneginf(loga) , np.isneginf(logb) ), x,  np.maximum(loga,logb) + np.log1p(np.exp( -abs(loga-logb) ) ) )        
        return e

    def mathlog0(self) :
        x = float('-inf')
        return np.array( [x] * self.dimensions ) 
        
    def mathlog1(self) :
        return np.array( [0] * self.dimensions )

    def logminusexp(self, loga, logb):
        return logb + np.log(np.exp(loga-logb)-1)

    def logprodexp(self, loga, logb):
        return loga + logb

    def logdivexp(self, loga, logb):
        return loga - logb

    def logpowexp(self, loga, exp):
        return loga * exp
    
    def log(self, x) :
        return np.log(x)
        
    def exp(self, x) :
        return np.exp(x)
        
class MathMath(object) :
    
    def __init__(self) :
        pass
        
    def logsumexp(self,loga, logb):
        if math.isinf(loga) and loga < 0 and math.isinf(logb) and logb < 0:
            return float("-inf")
        if logb < loga:
            t = loga
            loga = logb
            logb = t
        try:
            e = math.exp(loga-logb)
        except OverflowError as err:
            e = float("inf")
        return logb + math.log1p(e)

    def mathlog0(self) :
        return float('-inf')
        
    def mathlog1(self) :
        return 0

    def logminusexp(self, loga, logb):
        try:
            return logb + math.log(math.exp(loga-logb)-1)
        except ValueError as err:
            return self.mathlog0()

    def logprodexp(self, loga, logb):
        return loga + logb

    def logdivexp(self, loga, logb):
        return loga - logb

    def logpowexp(self, loga, exp):
        return loga * exp
    
    def log(self, x) :
        if x == 0 :
            return self.mathlog0()
        else :
            return math.log(x)
        
    def exp(self, x) :
        return math.exp(x)

rFirstLine = re.compile("""nnf ([0-9]+) ([0-9]+) ([0-9]+)""")
rLeaf      = re.compile("""L ([-]?[1-9][0-9]*)""")
rAnd       = re.compile("""A ([1-9][0-9]*) (.*)""")
rOr        = re.compile("""O ([0-9]+) ([1-9][0-9]*) (.*)""")
rTrue      = re.compile("""A 0""")
rFalse     = re.compile("""O [0-9]+ 0""")

class C2DAsAC:
    def __init__(self, c2dOutputFile, math):
        printv("Building C2DAsAc({})".format(c2dOutputFile))

        self.c2dOutputFile = c2dOutputFile
        self.math = math

    def probs(self, weightFunc):
        printv("Returning probs")
        vr, nbVars = self.values(weightFunc)
        dr = self.derivatives(vr, nbVars)
        Z = vr[-1]
        trueProbs = [self.math.exp(self.math.logdivexp(w, Z)) for w in dr]
        return self.math.exp(Z), trueProbs

    def values(self, weightFunc):
        """Traverse graph bottom to top to calculate partition function."""
        printv("Calculating values")
        printv("Reading nnf")
        src = open(self.c2dOutputFile, "r")
        firstLine = src.readline()
        mFirstLine = rFirstLine.match(firstLine)
        if mFirstLine == None:
            raise BaseException("{} did not parse.".format(firstLine))
        nbNodes = int(mFirstLine.group(1))
        nbVars = int(mFirstLine.group(3))
        vr = [0.0] * nbNodes
        lineIndex = 0

        def matchLine(line):
            mOr = rOr.search(line)
            if mOr != None:
                nbChildren = mOr.group(2)
                children = mOr.group(3)
                weights = [vr[child] for child in self.parseChildNodes(children)]
                weight = weights[0]
                for weightss in weights[1:]: weight = self.math.logsumexp(weight, weightss) 
                #print("OR weight = {}".format(weight))
                return weight

            mAnd = rAnd.search(line)
            if mAnd != None:
                nbChildren = mAnd.group(1)
                children = mAnd.group(2)
                weights = [vr[child] for child in self.parseChildNodes(children)]
                weight = weights[0]
                for weightss in weights[1:]: weight = self.math.logprodexp(weight, weightss)
                #print("AND weight = {}".format(weight))
                return weight

            mLeaf = rLeaf.search(line)
            if mLeaf != None:
                variable = int(mLeaf.group(1))
                if variable < 0:
                    lweight = weightFunc(-variable)[1]
                elif variable > 0:
                    lweight = weightFunc(variable)[0]
                else:
                    raise BaseException("Variable is 0")
                
                weight = self.math.log(lweight)
                
                #print("LEAF weight = {}".format(weight))
                return weight

            mFalse = rFalse.search(line)
            if mFalse != None:
                weight = self.math.mathlog0()
                #print("FALSE weight = {}".format(weight))
                return weight

            mTrue = rTrue.search(line)
            if mTrue != None:
                weight = self.math.mathlog1()
                #print("TRUE weight = {}".format(weight))
                return weight

            raise BaseException("{} did not parse.".format(line))
                
        for line in src.readlines():
            weight = matchLine(line)
            vr[lineIndex] = weight
            lineIndex += 1

        src.close()
        return vr, nbVars

    def derivatives(self, vr, nbVars):
        """Traverse graph from top to bottom to computer derivatives."""
        printv("Calculating derivatives")
        printv("Reading reversed nnf")
        src = open(self.c2dOutputFile+".reverse", "r")
        nbNodes = len(vr)
        dr = [self.math.mathlog0()] * nbNodes
        dr_indicator = [self.math.mathlog0()] * nbVars
        lineIndex = nbNodes - 1
        dr[lineIndex] = self.math.mathlog1()

        def matchLine(line):
            mOr = rOr.search(line)
            if mOr != None:
                nbChildren = mOr.group(2)
                children = mOr.group(3)
                childNodes = self.parseChildNodes(children)
                for child in childNodes:
                    dr[child] = self.math.logsumexp(dr[child], dr[lineIndex])
                       
                #print("OR")
                return

            mAnd = rAnd.search(line)
            if mAnd != None:
                nbChildren = mAnd.group(1)
                children = mAnd.group(2)
                childNodes = self.parseChildNodes(children)

                #totalprod = vr[childNodes[0]]
                #for vrss in (vr[cn] for cn in childNodes[1:]): totalprod = logprodexp(totalprod, vrss)

                for child in childNodes:
                    vrs = [vr[cn] for cn in childNodes if cn != child]
                    prod = vrs[0]
                    for vrss in vrs[1:]: prod = self.math.logprodexp(prod, vrss)
                    #prod = logdivexp(totalprod, vr[child])

                    dr[child] = self.math.logsumexp(dr[child], self.math.logprodexp(dr[lineIndex], prod))
                #print("AND")
                return

            mLeaf = rLeaf.search(line)
            if mLeaf != None:
                variable = int(mLeaf.group(1))
                if variable < 0:
                    pass # no op
                elif variable > 0:
                    # if (dr_indicator[variable-1] != self.math.mathlog0).any():
                    #     raise BaseException("Cannot have multiple leafs for identical literals, otherwise you get wrong results!")
                    dr_indicator[variable-1] = self.math.logprodexp(dr[lineIndex], vr[lineIndex])
                else:
                    raise BaseException("Variable 0 does not exist.")
                #print("LEAF")
                return

            mFalse = rFalse.search(line)
            if mFalse != None:
                return self.math.mathlog0()

            mTrue = rTrue.search(line)
            if mTrue != None:
                return self.math.mathlog0() # Different for derivative

            mFirstLine = rFirstLine.search(line)
            if mFirstLine != None:
                if int(mFirstLine.group(1)) != nbNodes:
                    raise BaseException("First line is inconsistent with node length")
                return

            raise BaseException("{} did not parse.".format(line))

        for line in src.readlines():
            matchLine(line)
            lineIndex -= 1

        src.close()
        return dr_indicator


    def parseChildNodes(self, line):
        return [int(part) for part in line.split(" ")]

def evaluate(ddnnf, weights, dims=None) :
    if not weights :
        return []
    else :
        
        if not dims :
            counter = C2DAsAC(ddnnf, MathMath())
            pEvidence, trueProbs = counter.probs(lambda variable: weights[variable])
        else :    
            counter = C2DAsAC(ddnnf, MathNP(dims))
            pEvidence, trueProbs = counter.probs(lambda variable: weights[variable])
        return trueProbs     

#! /usr/bin/env python

from __future__ import print_function
import sys

filename = 'zoo.dat'

line_parse = lambda line : line.strip().split(',')
with open(filename) as f :
    data = map(line_parse, f.readlines())
        
header = data[0]
data = data[1:]

out = sys.stdout
for row in data :
    for i, col_name in enumerate(header) :
        if col_name == 'id' :
            print ( 'animal(%s).' % (row[0]), file=out)
        elif col_name == 'has_legs' :
            if row[i] != '0' :
                print( '%s(%s,%s).' % (col_name, row[0], row[i]) , file=out)
        elif col_name == 'class' :
            print( '%s(%s).' % (row[i], row[0]) , file=out)
        else :
            if row[i] == '1' :
                print( '%s(%s).' % (col_name, row[0]) , file=out)
    print (file=out)

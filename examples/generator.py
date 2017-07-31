"""
Automatically generate large datasets containing common exceptions

"""
from __future__ import print_function

import argparse
import random

parser = argparse.ArgumentParser(description='Generate data for learning')
parser.add_argument('datafile', metavar='-o', help='file that data is written to')
parser.add_argument('settings', metavar='-s',
                    help='file to write settings to (modes and types)')
parser.add_argument('size', metavar='N', type=int, help='size of dataset')
#parser.add_argument('number of defaults', metavar='R', type=int,
#                    help='number of defaults to be learned')
#parser.add_argument('domain', metavar='D', help='categories')

args = parser.parse_args()

#data_file = open(args.datafile, 'w+')
#data_file.write('Hello')
#data_file.close()

birdlist = ['blackbird', 'sparrow', 'thrush', 'robin', 'eagle']
types = ['bird', 'penguin']+birdlist
print(types)

num = args.size+1
exceptions = int(round(num/10))+1
print(exceptions)

with open(args.datafile, 'w+') as w:
    for i in range(1, exceptions):
        w.write('bird(%d)\n' % i)
        w.write('penguin(%d)\n' % i)
        w.write('0.0::flies(%d)\n\n' % i)
    for i in range(exceptions, num):
        w.write('bird(%d)\n' % i)
        w.write('%s(%d)\n' % (random.choice(birdlist), i))
        w.write('flies(%d)\n\n' % i )

with open(args.settings, 'w+') as w:
    for t in types:
        w.write('mode(%s(+))\n' % t)
        w.write('base(%s(x))\n\n' % t)

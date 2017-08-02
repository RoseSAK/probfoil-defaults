"""
Automatically generate large datasets containing exceptions to general categorical
claims

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

birdlist = ['blackbird', 'sparrow', 'thrush', 'robin', 'eagle']
oddbirds = ['penguin', 'dodo', 'ostrich', 'kiwi'] # exceptions to the rule
nonbirds = ['cat', 'dog', 'rabbit'] # other types of things
types = ['bird']+birdlist+nonbirds+oddbirds

num = args.size+1 # size of dataset from user input
exceptions = int(round(num/10))+1 # take some as exceptions
others = int(round(num/10)) # take some as other things

# write data points to data file, randomly select from lists
with open(args.datafile, 'w+') as w: # create file if it does not exist already
    for i in range(1, exceptions):
        w.write('bird(%d).\n' % i)
        w.write('%s(%d).\n' % (random.choice(oddbirds),i))
        w.write('0.0::flies(%d).\n\n' % i)
    for i in range(exceptions, num-others):
        w.write('bird(%d).\n' % i)
        w.write('%s(%d).\n' % (random.choice(birdlist), i))
        w.write('flies(%d).\n\n' % i )
    for i in range(num-others, num):
        w.write('0.0::bird(%d).\n' % i)
        w.write('%s(%d).\n' % (random.choice(nonbirds), i))
        w.write('0.0::flies(%d).\n\n' % i )

# write settings (types and modes) to settings file, for all predicates
with open(args.settings, 'w+') as w:
    for t in types:
        w.write('mode(%s(+)).\n' % t)
        w.write('base(%s(x)).\n\n' % t)
    w.write('base(flies(x)).\n')
    w.write('learn(flies/1).\n') # specify target predicate
    w.write('example_mode(closed).\n\n') # use only examples in the data

#! /usr/bin/env python3

import sys
import os

def main(filenames) :
    
    for filename_in in filenames :
        filename_out = os.path.splitext(filename_in)[0] + '.pl'
        with open(filename_in) as file_in :
            with open(filename_out, 'w') as file_out :
                line_num = 0
                for line_in in file_in :
                    line_in = line_in.strip()
                    if line_in and not line_in.startswith('@') :
                        values = list(map(float,line_in.split(',')))
                        num_atts = len(values)
                        line_out = '\n'.join( '%s::att%s(%s).' % (val, att, line_num) for att, val in enumerate(values) ) + '\n\n'
                        file_out.write(line_out)
                        line_num += 1
                line_out = '\n'.join( 'base(att%s(id)).' % att for att in range(0, num_atts) ) + '\n\n'
                file_out.write(line_out)    
        
if __name__ == '__main__' :
    main(sys.argv[1:])
    

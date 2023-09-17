#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':

    os.chdir(os.path.dirname(__file__))
    print(os.getcwd())
    os.environ['image_path'] = sys.argv[1]
    print(sys.argv[1])
    os.system('matlab -nodisplay -nodesktop -nosplash -r RBCNet_RunMe')
    


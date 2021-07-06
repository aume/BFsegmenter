#! /usr/bin/python
'''
downloadconvert_htn.py
Miles Thorogood
October 2012

'''

import urllib
import os
import subprocess


verbose = True

class converter:

    def __init__(self) :
        self.out_name = ''
        
    # path is where to look at put files
    # file will be file.suffix
    # format aiff wav etc.
    def convert(self, path, file, out_format) :
        print(path,file,out_format)
        in_suffix = os.path.splitext(file)[1]
        print in_suffix
        temp = file.split(in_suffix,1)
        print '02'
        self.out_name = out_format.join(temp)
        
        noError=True #
	        
    	if os.path.isfile(path+self.out_name):
    		if verbose: print "File previously retrieved"
    		else: None
        else:
            try:
                cmd ='sox %s -b 16 -c 1 -r 44100 %s' % (path+file, path+self.out_name) # requires SOX
                subprocess.call(cmd, shell=True)
                print 'converted'
            except:
                print 'baddness in converting'
                pass
						
			
	

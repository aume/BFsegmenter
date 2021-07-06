#!/usr/bin/env python
#
# processSingle.py
# 
# processes a single audio file (given at the cmd line) to output 
# start, end, bfclass, valence, arousal
#
# Miles Thorogood 2014
# mthorogo@sfu.ca
# http://www.audiometaphor.ca
#


from regionExtract import extract_regions
import afconvert  
import os, sys
import csv


    
segmenter = extract_regions() # region extraction

                   
fileInfo = os.path.split(sys.argv[1])

sound = {"path":fileInfo[0], "filename":fileInfo[1]}
print sound
                        
dir_path = sound["path"]    
file_name = sound["filename"]
filewithpath = os.path.join(dir_path, file_name)

regions = []

yay = True
try:
    regions = segmenter.process_file(filewithpath) # extract the region data
    #print regions
except:
    print "something happened in audio_crawler local"
    yay = False
if yay: # skip it if broke
#duration = sound['duration']
    print "YAY"
    with open(file_name+".csv", "wb") as f:
        writer = csv.writer(f,delimiter=',')
        for region in regions:
            writeMe =  region['type'], region['start'], region['end'], region['duration'], region['valence'], region['arousal']
            writer.writerow(writeMe)
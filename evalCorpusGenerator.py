#!/usr/bin/python

'''
segCorpusGenerator.py

A little program to generate a corpus of concatenated bf segments.
Segments are sourced from a directory structure of the following layout:
back fore bofo

A new corpus item is a single combinatorial concatenation
B->F
F->B
BF->F
BF->B
F->BF
B->BF

Program:
* First create a dictionary for background and foregroung and bfground
Go through the directory of the directory structure
Add a new key to the dictionary for the directory name, with an array value
Fill the array with the absolute path for each audio file in the directory
* The next stage will run the combinatorial conjuction of items in the dictionary
'''


from pydub import AudioSegment
import sys
import os
from itertools import permutations
from random import choice
import csv 
segs = {}
for dirname, dirnames, filenames in os.walk(sys.argv[1]):
    #print dirname
    for subdirname in dirnames:
        segs[subdirname]=[]
        validir = os.path.join(dirname, subdirname)
        for file in os.listdir(validir):
            if os.path.splitext(file) [1] == '.aiff':
                #print (validir,file)
                segs[subdirname].append(validir+'/'+file)
                
                
with open('evalCorpus/'+'groundTruth'+'.csv', 'wb') as csvfile:
    segmentwriter = csv.writer(csvfile, delimiter=',')
    for p in permutations(segs): 
        for i in range(int(sys.argv[2])):
            empty = AudioSegment.empty()
            filename = str(p).replace("'", '').replace('(', '').replace(')', '').replace(',', '-').replace(' ', '')
            toCSV = [filename+'-'+str(i)+'.wav']
            timeAccum = 0.0
            for item in p*3:
                concatAudio = AudioSegment.from_file(choice(segs[item]))
                duration_in_milliseconds = len(concatAudio)
                empty = empty.append(concatAudio,crossfade=0)
                toCSV.append(item)
                toCSV.append(timeAccum)
                timeAccum += duration_in_milliseconds/1000.0
                toCSV.append(timeAccum)
                print choice(segs[item])
            segmentwriter.writerow(toCSV)
            empty.export('evalCorpus/'+filename+'-'+str(i)+'.wav', format="wav")
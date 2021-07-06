#!/usr/bin/python


'''
This script creates the segmentation algorithms evaluation data

The program runs the algorithms one after the other on the same corpus located in sys.argv[1]
For each file in the corpus, the alogithm inserts segmentation points and class labels 
the insertion points and labels are compared to the ground truth data inferred in the file name.
File names contain the class labels for each equal length concatentated segment. 
So, if there is n labels, then ground truth insertion points for those labels with 
be at multiples of duration/n starting at 0 and ending at n-1

A sv file is created with the insertion and class types.

The error rate of the algorithms is computed as the difference between the insertion point and the ground truth.
The sum of the errors is then divided by the total time of the signal, observed as the mean error rate by unit of time.
galliano2009ester


'''

import regionExtract
import csv
import sys
from copy import deepcopy

extractor = regionExtract.extract_regions()

#
segmenters = ['median', 'k-depth', 'max_posterior']

# keep track of [{'start':start, 'end':end},...]
logging = {'fore':[], 'back':[], 'backfore':[]}
 
# for each algorithm keep track of how long a class is 'detected', how long a class is 'actual'ly there, and 'true-positive' 
metrics = {'detected':0, 'actual':0, 'true-positve':0, 'false-positve':0, 'false-negative':0, 'precision':0, 'recall':0, 'f-measure':0}
#statistics = {'fore':deepcopy(metrics), 'back':deepcopy(metrics), 'backfore':deepcopy(metrics)}
algorithms = {}
for algorithm in segmenters:
    algorithms[algorithm] = {'fore':deepcopy(metrics), 'back':deepcopy(metrics), 'backfore':deepcopy(metrics)}

    
def checkOverlap(seg1, seg2):
    start = max(seg1['start'],seg2['start'])    
    end = min(seg1['end'],seg2['end'])
    if start<end:
        return end-start
    else:
        return 0 
    
with open(sys.argv[1], 'rb') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for item in datareader: # for each file

        # get the ground truth data for this clip
        # ground truth
        truth = deepcopy(logging)
        #actual = 0
        for i in xrange(1, len(item), 3):
            truth[item[i]].append({'start':float(item[i+1]), 'end':float(item[i+2])})
            #actual += float(item[i+2])-float(item[i+1]) # ground truth
            
        
        # for each algorithm
        for algorithm in algorithms:
            print('starting '+ algorithm)
            
            inserted = deepcopy(logging)
            segments = extractor.process_file('evalCorpus/'+item[0], algorithm)
            
            # detected events
            #print segments

            with open(segments[0]+'-'+algorithm+'.csv', 'wb') as csvfile:
                segmentwriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)     
      
                for seg in segments[1]: # because segments[0] is the filename
                    inserted[seg['type']].append({'start':float(seg['start']), 'end':float(seg['start'])+float(seg['duration'])})
                    algorithms[algorithm][seg['type']]['detected'] += float(seg['duration']) # detected events
                
                    segmentwriter.writerow([seg['start'],seg['duration'],seg['type']])
                
            # true poistives
            for bf in truth:                
                for seg1 in truth[bf]:
                    algorithms[algorithm][bf]['actual'] += float(seg1['end'])-float(seg1['start']) # ground truth
                    
                    for seg2 in inserted[bf]:
                        algorithms[algorithm][bf]['true-positve'] += checkOverlap(seg1,seg2) # true poistives
            

    

def precision(trueP, falseP):
    if trueP == 0:
        return 0
    return trueP / (trueP + falseP)
    
def recall(trueP, falseN):
    if trueP == 0:
        return 0
    return trueP / (trueP + falseN)
    
def fMeaure(p, r):
    if p<=0 and r <=0:
        return 0
    else:
        return (2*p*r)/p+r
    
for algorithm in algorithms:
    print algorithm
    for bftype in algorithms[algorithm]:
        item = algorithms[algorithm][bftype]
        print '\t'+bftype
        detected = item['detected']
        actual = item['actual']
        trueP = item['true-positve']
        
        item['false-positve'] = item['detected']-item['true-positve']
        item['false-negative'] = item['actual']-item['true-positve']
        
        item['precision'] = precision(trueP, item['false-negative'])
        item['recall'] = recall(trueP, item['false-positve'])
        
        item['f-measure'] = fMeaure(item['precision'], item['recall'])
        metrics = ['precision', 'recall', 'f-measure']
        #print'\t\t'+'f-measure: '+str(item['f-measure'])
        for metric in metrics:
            print '\t\t'+metric + ': '+str(algorithms[algorithm][bftype][metric])





# go through the directory
# for file in os.listdir(validir):
#     #make sure it is aiff
#     if os.path.splitext(file) [1] == '.aiff':
#         #extract the info from the fle name
#         validir+'/'+file
#!/usr/bin/python
'''
Miles Thorogood
mthorogo@sfu.ca
Nov 2013


corpusAquire. Script to segment and cut an audio file. 
Cut file is saved to disk and results logged in sql db. 
'''
from svnsegmenter import Segmenter
import sys
import os
import sys
import csv



class extract_regions:
    
    def __init__(self):
        # init the segmenter and train with BF Corpus
        self.segmenter = Segmenter('./feature_output.txt')
        #self.process_file(file)

    def process_file(self, file, algorithm='median', k=3):
        
        rli = [file]
        
        if algorithm == 'median':
            processed = self.segmenter.regionsChunk(file) # segment the file
            processed = self.segmenter.smoothProbabilities(processed,7)
            processed = self.segmenter.conjunction(processed)
            rli.append(processed)
            
        elif algorithm == 'max_posterior':
            processed = self.segmenter.regionsChunk(file) # segment the file
            #processed = self.segmenter.smoothProbabilities(processed,7)
            processed = self.segmenter.max_posterior(processed, 4)
            processed = self.segmenter.conjunction(processed)
            rli.append(processed)
            
        elif algorithm == 'k-depth':            
            processed = self.segmenter.regionsChunk(file) # segment the file
            #processed = self.segmenter.smoothProbabilities(processed,7)
            processed = self.segmenter.simple_k(processed, 7)
            #processed = self.segmenter.simple_median(processed, 3)
            processed = self.segmenter.conjunction(processed)
            rli.append(processed)
            
        elif algorithm == 'k-reconcile':
            processed = self.segmenter.regionsSlide(file, 20) # segment the file
            processed = self.segmenter.smoothProbabilities(processed,31)
            processed = self.segmenter.conjunction(processed)
            rli.append(processed)
        return rli
        






def main():
    extract_regions(sys.argv[1])
    sys.exit(0)
    
if __name__ == "__main__":
    main()
    

   
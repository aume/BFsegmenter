#! /usr/bin/python

'''
svnsegmenter.py
Miles Thorogood
October 2015

'''

import sys
import csv
from math import sqrt, floor
#from sklearn.datasets import load_svmlight_file
#from sklearn import svm
import bf_classifier
import affect_predictor
from yaafeEngine import yaafengine
import numpy as np
from scipy import ndimage

# model data is in svm format derived from yaafeDirSVM.py
# train model
# segment new files

debug = False


class Segmenter:
    # initialize with training data
    def __init__(self, training_data):
        
        # create the models
        self.clf = bf_classifier.BFClassifier()
        self.afp = affect_predictor.AffectPredict()
        # the yaafe engine for extracting features
        # Our yaafe engine make sure that the features were extracted under the same conditions as the training data
        self.window_size = 0.5 # analysis window length in seconds
        self.sRate = 22500 # sample rate 
        self.nsamps = 1024 # samples in each frame
        self.yengine = yaafengine(self.sRate, self.nsamps)
	
	
	# data will be [[file, file],... , [file, file]]
	#returns [ [file, [regions]], ..., [file, [regions]] ]
    def process_data(self, data):
        out_data = []
        for result in data:
            result_data = []
            for file in result:
                result_data.append(self.extract(file)) # adds [file_path, [['type', start, end], [...], ['type'n, startn, endn]]]
            out_data.append(result_data) # adds [[file,[regions]],file,[regions]]]
    	return out_data

		



    # audio file path
    # returns # [file_path, [['type', start, end], [...], ['type'n, startn, endn]]]
    def regionsChunk(self, afile):        
        fvecs,fnames = self.yengine.featureVectors(afile)
        #print fnames
        featnames = []
        for name in fnames:
            featnames.append(name + '_mean')
            featnames.append(name + '_std')
        # calculate the length of analysis frames
        frame_duration = float(self.nsamps/2)/float(self.sRate)
        num_frames = int(self.window_size / frame_duration) # number frames in a window
        # dictionary for class names from libsvm format
        types = {1:'back', 2:'fore', 3:'backfore'}

        '''
            Segmentation
        '''
        if debug: print 'Begin segmentation'
        processed = [] # storage for the classified segments
        # scan across by non-overlapping windows
        for i in xrange(0, len(fvecs), num_frames):
            window = fvecs[i:i+num_frames] # get the vectors for the window
            accum = zip(*window)
            features = []
            # Do the BOF
            for cc in accum: # go through each dimension
                n = len(cc)
                mean = sum(cc)/n
                std = sqrt(sum((x-mean)**2 for x in cc) / n)

                mvalue = mean
                features.append(mvalue)
                svalue = std
                features.append(svalue)
            
            featdic = {}
            for name,val in zip(featnames,features):
                featdic[name] = val 
            # predict the class
            vect = [featdic['Loudness_mean'], featdic['Loudness_std'], featdic['MFCC1_mean'], featdic['MFCC1_std'], featdic['MFCC2_mean'], featdic['MFCC2_std'], featdic['MFCC3_mean'], featdic['MFCC3_std']]
            
            typ = types[int(self.clf.predict(vect))]
            prob = self.clf.predictProb(vect)
            #print(typ,prob)
            
            start_time = float(i*(self.nsamps/2))/float(self.sRate)
            end_time = float((i+num_frames)*(self.nsamps/2))/float(self.sRate)
            processed.append({'type':typ, 'probabilities':prob, 'start':start_time, 'end':end_time, 'feats':featdic, 'count':1})
            #print (start_time, end_time, type)
        if debug: print 'Finished segmentation'
        return processed
    
    #
    # same as above but slides by ms slidetime instead of window chunks
    #
    def regionsSlide(self, afile, slidetime=20.0):        
        fvecs,fnames = self.yengine.featureVectors(afile)
        #print fnames
        featnames = []
        for name in fnames:
            featnames.append(name + '_mean')
            featnames.append(name + '_std')
        # calculate the length of analysis frames
        frame_duration = float(self.nsamps/2)/float(self.sRate)
        num_frames = int(self.window_size / frame_duration) # number frames in a window
        
        frame_slide = max(1, round((slidetime*0.001)/frame_duration)) # how many frames to slide alonf
        # dictionary for class names from libsvm format
        types = {1:'back', 2:'fore', 3:'backfore'}

        '''
            Segmentation
        '''
        if debug: print 'Begin segmentation'
        processed = [] # storage for the classified segments
        # scan across by window length
        for i in xrange(0, len(fvecs), int(frame_slide)):
            window = fvecs[i:i+num_frames] # get the vectors for the window
            accum = zip(*window)
            features = []
            # Do the BOF
            for cc in accum: # go through each dimension
                n = len(cc)
                mean = sum(cc)/n
                std = sqrt(sum((x-mean)**2 for x in cc) / n)

                mvalue = mean
                features.append(mvalue)
                svalue = std
                features.append(svalue)
            
            featdic = {}
            for name,val in zip(featnames,features):
                featdic[name] = val 
            # predict the class
            vect = [featdic['Loudness_mean'], featdic['Loudness_std'], featdic['MFCC1_mean'], featdic['MFCC1_std'], featdic['MFCC2_mean'], featdic['MFCC2_std'], featdic['MFCC3_mean'], featdic['MFCC3_std']]
            typ = types[int(self.clf.predict(vect))]
            prob = self.clf.predictProb(vect)
            
            start_time = float(i*(self.nsamps/2))/float(self.sRate)
            end_time = float((i+num_frames)*(self.nsamps/2))/float(self.sRate)
            processed.append({'type':typ, 'probabilities':prob, 'start':start_time, 'end':end_time, 'feats':featdic, 'count':1})
            #print (start_time, end_time, type)

        if debug: print 'Finished segmentation'
        return processed
    
    def smoothProbabilities(self, processed, winSize=3):
        a = np.array(processed[0]['probabilities'])
        
        for i in xrange(0,len(processed),1):
            #print processed[i]['probabilities']
            a = np.vstack((a,processed[i]['probabilities']))
        #print a
        
        for i in xrange(a.shape[1]):
            a[:,i]= ndimage.filters.median_filter(a[:, i], size=winSize)
        #print a

        import operator
        labels = ['back', 'fore', 'backfore']
        for i in xrange(0,len(processed),1):
            processed[i]['probabilities'] = a[i]
            index, value = max(enumerate(a[i]), key=operator.itemgetter(1))
            processed[i]['type']=labels[index]
        return processed
        
    #
    #
    # a simple k-depth filtering
    #         
    def simple_k(self, processed, k_depth):
        # conjunction (renaming segments) giving preference to foreground
        start = 0
        while start < len(processed):
            # If we have a fg
            #print start                
            if processed[start]['type'] != '': # don't join background sounds
                trigger_type = processed[start]['type'] # log type
                
                for i in xrange(start+k_depth, start-1, -1):
                    if i < len(processed):
                        if processed[i]['type'] == trigger_type:
                            for j in xrange(start, i,1): 
                                processed[j]['type'] = trigger_type
                                start = i
                                pass
            start +=1
       
        if debug: print 'Finished k-depth mrking'
        return processed
            
    
    #
    # a simple median filtering
    #
    def max_posterior(self, processed, m_span=7):
        import operator
        medWin = m_span#int(floor(m_span/2))
        filtered = []
        for i in xrange(0,len(processed),1):
            #print(i,'old',processed[i]['type'])
            #if processed[i]['type'] != processed[i+1]['type']:
            labels={'back':0, 'fore':0, 'backfore':0}
            for j in xrange(max(0,i-medWin), min(i+medWin,len(processed)), 1):
                k = processed[j]['type']
                labels[k] = labels.setdefault(k, 0) + 1
            maxlabel = max(labels.iteritems(), key=operator.itemgetter(1))[0]
            filtered.append(maxlabel)

        for i in xrange(0,len(processed),1):
            #print (i,'old',processed[i]['type'])
            if processed[i]['type'] != filtered[i]:
               processed[i]['type'] = filtered[i]
               #print (i,'change',processed[i]['type'])

        return processed
            
        
        #
        # k-reconcile
        #    
    def k_reconcile(self, processed):
        
        return processed
    
    def conjunction(self, processed):
        # Here we join up any same labelled adjacent regions 
        for i in xrange(1, len(processed), 1):
            if processed[i]['type'] == processed[i-1]['type']: # if its the same
                processed[i]['start'] = processed[i-1]['start'] # update the start time
                processed[i]['feats'] =  self.sumFeatureDics(processed[i]['feats'], processed[i-1]['feats'])
                processed[i]['count'] += processed[i-1]['count']
                processed[i-1]['type'] = 'none' # nullify the previos segment
                #print processed[i]['type']
            else:
                pass
        if debug: print 'Finished conjunction'
        return self.finalize_regions(processed)
        
    def finalize_regions(self, processed):
        if debug: print 'Begin Logging'
        file_data = []#[afile] # [file_path, [['type', start, end], [...], ['type'n, startn, endn]]]
        region_data = []
        for i in processed:
            if i['type'] is not 'none':
                temp = {}
                temp['type'] = i['type']
                temp['duration'] = i['end'] - i['start'] # duration
                temp['start'] = i['start']
                temp['end'] = i['end']
                temp['feats'] = self.avgDicItems(i['feats'], i['count'])
                f = temp['feats']
                vect = [f['Loudness_mean'], f['Loudness_std'], f['MFCC1_mean'], f['MFCC1_std'], f['MFCC2_mean'], f['MFCC2_std'], f['MFCC3_mean'], f['MFCC3_std']]
                temp['valence'] = self.afp.predict_valence(vect)
                temp['arousal'] = self.afp.predict_arousal(vect)
                region_data.append(temp)
		file_data.append(region_data)
        if debug: print 'End Logging'
        #print 'length of file data is ', len(file_data)
        return region_data
	
    def avgDicItems(self,D,a):
        result = {}
        for key in D.keys():
            result[key] = D[key]/a
        return result
        
    def sumFeatureDics(self, Da, Db):
        result = {}
        for key in Da.keys():
            A = Da[key]
            B = Db[key]
            result[key] = A+B#[a + b for (a,b) in zip(A,B)]
        return result
    
        
        
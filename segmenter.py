import sys
import csv
from math import sqrt, floor
from EssentiaEngine import EssentiaEngine
import bf_classifier
import affect_predictor
import numpy as np
from scipy import ndimage

from essentia.standard import MonoLoader, FrameGenerator, PoolAggregator
import essentia.utils as utils
import essentia 

# model data is in svm format derived from yaafeDirSVM.py
# train model
# segment new files

debug = False


class Segmenter:
    # initialize with training data
    def __init__(self, training_data):
        
        # create the models TODO
        self.clf = bf_classifier.BFClassifier()
        #self.afp = affect_predictor.AffectPredict()

        # the yaafe engine for extracting features
        # Our yaafe engine make sure that the features were extracted under the same conditions as the training data
        self.windowDuration = 0.5 # analysis window length in seconds
        self.sampleRate = 44100 # sample rate 
        self.frameSize = 1024 # samples in each frame
        self.hopSize = 512
        self.engine = EssentiaEngine(self.sRate, self.nsamps)
	
	
	# data will be [[file, file],... , [file, file]] TODO
	#returns [ [file, [regions]], ..., [file, [regions]] ]
    # def process_data(self, data):
    #     out_data = []
    #     for result in data:
    #         result_data = []
    #         for file in result:
    #             result_data.append(self.extract(file)) # adds [file_path, [['type', start, end], [...], ['type'n, startn, endn]]]
    #         out_data.append(result_data) # adds [[file,[regions]],file,[regions]]]
    # 	return out_data

		
    # audio file path
    # returns # [file_path, [['type', start, end], [...], ['type'n, startn, endn]]]
    def regionsChunk(self, afile):        

        # instantiate the loading algorithm
        loader = MonoLoader(filename = afile, sampleRate = self.sampleRate)
        # perform the loading
        audio = loader()

        # create pool for storage and aggregation
        pool = essentia.Pool()

        # frame counter used to detect end of window
        frameCount_window = 0
        frameCount_file = 0

        # calculate the length of analysis frames
        frame_duration = float(self.frameSize / 2)/float(self.sampleRate)
        numFrames_window = int(self.windowDuration / frame_duration) # number frames in a window

        # dictionary for class names from libsvm format
        types = {1:'back', 2:'fore', 3:'backfore'}

        processed = [] # storage for the classified segments
        
        for frame in FrameGenerator(audio, frameSize=self.frameSize, hopSize=self.hopSize, startFromZero=True, lastFrameToEndOfFile = True):
            # spectral contrast valleys
            frame_windowed = self.engine.window(frame)
            frame_spectrum = self.engine.spectrum(frame_windowed)
            sc_valley = self.engine.spectral_contrast(frame_spectrum)
            pool.add('lowlevel.spectral_contrast_valleys', sc_valley)

            # silence rate
            pool.add('lowlevel.silence_rate', self.engine.silence_rate(frame))

            # spectral flux
            pool.add('lowlevel.spectral_flux', self.engine.spectral_flux(frame_spectrum))

            # Gammatone-frequency cepstral coefficients
            gfccs = self.engine.gfcc(frame_spectrum)
            pool.add('lowlevel.gfcc', gfccs)

            # spectral RMS
            pool.add('lowlevel.spectral_rms', self.engine.rms(frame_spectrum))

            # increment counters
            frameCount_window += 1
            frameCount_file += 1

            # detect if we have traversed a whole window
            if (frameCount_window == numFrames_window):
                # compute mean and variance of the frames
                aggrPool = PoolAggregator(defaultStats = [ 'mean', 'stdev' ])(pool)
                features_dict = {}
                features_dict['lowlevel.silence_rate.stdev'] = aggrPool['lowlevel.silence_rate.stdev']
                features_dict['lowlevel.spectral_contrast_valleys.mean.0'] = aggrPool['lowlevel.spectral_contrast_valleys.mean'][0]
                features_dict['replay_gain'] = pool['replay_gain'][0]
                features_dict['lowlevel.spectral_contrast_valleys.stdev.2'] = aggrPool['lowlevel.spectral_contrast_valleys.stdev'][2]
                features_dict['lowlevel.spectral_contrast_valleys.stdev.3'] = aggrPool['lowlevel.spectral_contrast_valleys.stdev'][3]
                features_dict['lowlevel.spectral_contrast_valleys.stdev.4'] = aggrPool['lowlevel.spectral_contrast_valleys.stdev'][4]
                features_dict['lowlevel.spectral_contrast_valleys.stdev.5'] = aggrPool['lowlevel.spectral_contrast_valleys.stdev'][5]
                features_dict['lowlevel.spectral_flux.mean'] = aggrPool['lowlevel.spectral_rms.mean']
                features_dict['lowlevel.gfcc.mean.0'] = aggrPool['lowlevel.gfcc.mean'][0]
                features_dict['lowlevel.spectral_rms.mean'] = aggrPool['lowlevel.spectral_rms.mean']

                # replay gain 
                pool.add('replay_gain', self.engine.rgain(audio[frameCount_file : frameCount_file + frameCount_window]))

                # reset counter and clear pool
                frameCount_window = 0
                pool.clear()
                aggrPool.clear()

                # prepare feature values to predict the class
                vect = features_dict.values()

                type = types[int(self.clf.predict(vect))]
                prob = self.clf.predictProb(vect)
                print(features_dict)
                start_time = float(frameCount_file*(self.frameSize/2))/float(self.sampleRate)
                end_time = float((frameCount_file+numFrames_window)*(self.frameSize/2))/float(self.sampleRate)
                processed.append({'type':type, 'probabilities':prob, 'start':start_time, 'end':end_time, 'feats':features_dict, 'count':1})
        

        return processed
    

    
    # def smoothProbabilities(self, processed, winSize=3):
    #     a = np.array(processed[0]['probabilities'])
        
    #     for i in xrange(0,len(processed),1):
    #         #print processed[i]['probabilities']
    #         a = np.vstack((a,processed[i]['probabilities']))
    #     #print a
        
    #     for i in xrange(a.shape[1]):
    #         a[:,i]= ndimage.filters.median_filter(a[:, i], size=winSize)
    #     #print a

    #     import operator
    #     labels = ['back', 'fore', 'backfore']
    #     for i in xrange(0,len(processed),1):
    #         processed[i]['probabilities'] = a[i]
    #         index, value = max(enumerate(a[i]), key=operator.itemgetter(1))
    #         processed[i]['type']=labels[index]
    #     return processed
        
    # #
    # #
    # # a simple k-depth filtering
    # #         
    # def simple_k(self, processed, k_depth):
    #     # conjunction (renaming segments) giving preference to foreground
    #     start = 0
    #     while start < len(processed):
    #         # If we have a fg
    #         #print start                
    #         if processed[start]['type'] != '': # don't join background sounds
    #             trigger_type = processed[start]['type'] # log type
                
    #             for i in xrange(start+k_depth, start-1, -1):
    #                 if i < len(processed):
    #                     if processed[i]['type'] == trigger_type:
    #                         for j in xrange(start, i,1): 
    #                             processed[j]['type'] = trigger_type
    #                             start = i
    #                             pass
    #         start +=1
       
    #     if debug: print 'Finished k-depth mrking'
    #     return processed
            
    
    # #
    # # a simple median filtering
    # #
    # def max_posterior(self, processed, m_span=7):
    #     import operator
    #     medWin = m_span#int(floor(m_span/2))
    #     filtered = []
    #     for i in xrange(0,len(processed),1):
    #         #print(i,'old',processed[i]['type'])
    #         #if processed[i]['type'] != processed[i+1]['type']:
    #         labels={'back':0, 'fore':0, 'backfore':0}
    #         for j in xrange(max(0,i-medWin), min(i+medWin,len(processed)), 1):
    #             k = processed[j]['type']
    #             labels[k] = labels.setdefault(k, 0) + 1
    #         maxlabel = max(labels.iteritems(), key=operator.itemgetter(1))[0]
    #         filtered.append(maxlabel)

    #     for i in xrange(0,len(processed),1):
    #         #print (i,'old',processed[i]['type'])
    #         if processed[i]['type'] != filtered[i]:
    #            processed[i]['type'] = filtered[i]
    #            #print (i,'change',processed[i]['type'])

    #     return processed
            
        
    #     #
    #     # k-reconcile
    #     #    
    # def k_reconcile(self, processed):
        
    #     return processed
    
    # def conjunction(self, processed):
    #     # Here we join up any same labelled adjacent regions 
    #     for i in xrange(1, len(processed), 1):
    #         if processed[i]['type'] == processed[i-1]['type']: # if its the same
    #             processed[i]['start'] = processed[i-1]['start'] # update the start time
    #             processed[i]['feats'] =  self.sumFeatureDics(processed[i]['feats'], processed[i-1]['feats'])
    #             processed[i]['count'] += processed[i-1]['count']
    #             processed[i-1]['type'] = 'none' # nullify the previos segment
    #             #print processed[i]['type']
    #         else:
    #             pass
    #     if debug: print 'Finished conjunction'
    #     return self.finalize_regions(processed)
        
    # def finalize_regions(self, processed):
    #     if debug: print 'Begin Logging'
    #     file_data = []#[afile] # [file_path, [['type', start, end], [...], ['type'n, startn, endn]]]
    #     region_data = []
    #     for i in processed:
    #         if i['type'] is not 'none':
    #             temp = {}
    #             temp['type'] = i['type']
    #             temp['duration'] = i['end'] - i['start'] # duration
    #             temp['start'] = i['start']
    #             temp['end'] = i['end']
    #             temp['feats'] = self.avgDicItems(i['feats'], i['count'])
    #             f = temp['feats']
    #             vect = [f['Loudness_mean'], f['Loudness_std'], f['MFCC1_mean'], f['MFCC1_std'], f['MFCC2_mean'], f['MFCC2_std'], f['MFCC3_mean'], f['MFCC3_std']]
    #             temp['valence'] = self.afp.predict_valence(vect)
    #             temp['arousal'] = self.afp.predict_arousal(vect)
    #             region_data.append(temp)
	# 	file_data.append(region_data)
    #     if debug: print 'End Logging'
    #     #print 'length of file data is ', len(file_data)
    #     return region_data
	
    # def avgDicItems(self,D,a):
    #     result = {}
    #     for key in D.keys():
    #         result[key] = D[key]/a
    #     return result
        
    # def sumFeatureDics(self, Da, Db):
    #     result = {}
    #     for key in Da.keys():
    #         A = Da[key]
    #         B = Db[key]
    #         result[key] = A+B#[a + b for (a,b) in zip(A,B)]
    #     return result
    
        
        
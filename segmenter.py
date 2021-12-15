from math import sqrt, floor
from essentia_engine import EssentiaEngine
import bf_classifier
import affect_predictor
import numpy as np
from scipy import ndimage

from essentia.standard import MonoLoader, FrameGenerator, PoolAggregator
import essentia.utils as utils
import essentia

debug = False

# processfile?

class Segmenter:
    # initialize
    def __init__(self):

        # create the models
        self.clf = bf_classifier.BFClassifier()
        self.afp = affect_predictor.AffectPredict()

        self.windowDuration = 1.5 # analysis window length in seconds
        self.sampleRate = 44100  # sample rate
        self.frameSize = 2048  # samples in each frame
        self.hopSize = 1024
        self.windowSize = int(self.sampleRate * self.windowDuration)
        self.adjustedWindow = (self.windowSize // self.frameSize) * self.frameSize

        self.smoothing_window = 1
        self.medianFilter_span = 3

        self.filterWindow = 3

        # the essentia engine make sure that the features were extracted under the same conditions as the training data
        self.engine = EssentiaEngine(self.sampleRate, self.frameSize, self.hopSize)

    # run the segmentation
    def segment(self, afile):
        segments = self.extractRegions(afile)

        # segment filtering
        # segments = self.marginSmoothing(segments)

        segments = self.foregroundExpansion(segments)
        segments = self.foregroundClustering(segments)

        # segments = self.smoothProbabilities(segments, self.smoothing_window)
        # segments = self.max_posterior(segments, self.medianFilter_span)

        segments = self.medianFiltering(segments)

        segments = self.foregroundClustering(segments)


        # join segments
        segments = self.conjunction(segments)
        return segments

    # use the bf classifier to extract background, foreground, bafoground regions
    # returns # [file_path, [['type', start, end], [...], ['type'n, startn, endn]]]
    def extractRegions(self, afile):

        # instantiate the loading algorithm
        loader = MonoLoader(filename=afile, sampleRate=self.sampleRate)
        # perform the loading
        audio = loader()

        # create pool for storage and aggregation
        pool = essentia.Pool()

        # frame counter used to detect end of window
        windowCount = 0

        # calculate the length of analysis frames
        frame_duration = float(self.frameSize / 2)/float(self.sampleRate)
        
        # number frames in a window
        numFrames_window = int(self.windowDuration / frame_duration)

        print(numFrames_window, ' frames in a window')
        print('frame duration: ', frame_duration)
        print('audio len: ', len(audio))
        print('number frames total: ', len(audio)/self.frameSize)
        print('window size: ', self.windowSize)
        print('frame adjusted window size: ', self.adjustedWindow)

        # translate type naming convention from csv to database 
        types = {1: 'fore', 2: 'back', 3: 'backfore'}

        processed = []  # storage for the classified segments

        for window in FrameGenerator(audio, frameSize=self.adjustedWindow, hopSize=self.adjustedWindow, startFromZero=True, lastFrameToEndOfFile=True):
            # extract all features
            pool = self.engine.extractor(window)
            aggrigatedPool = essentia.standard.PoolAggregator(defaultStats=['mean', 'stdev', 'skew', 'dmean', 'dvar', 'dmean2', 'dvar2'])(pool)

            # compute mean and variance of the frames using the pool aggregator, assign to dict in same order as training
            # narrow everything down to select features
            features_dict = {}
            descriptorNames = aggrigatedPool.descriptorNames()

            # unpack features in lists 
            for descriptor in descriptorNames:
                if('tonal' in descriptor or 'rhythm' in descriptor):
                    continue
                value = aggrigatedPool[descriptor]
                if (str(type(value)) == "<class 'numpy.ndarray'>"):
                    for idx, subVal in enumerate(value):
                        features_dict[descriptor + '.' + str(idx)] = subVal
                    continue
                else:
                    if(isinstance(value,str)):
                        pass
                    else:
                        features_dict[descriptor] = value

            # filter features for bafo prediction TODO

            # reset counter and clear pool
            pool.clear()
            aggrigatedPool.clear()

            # prepare feature values to predict the class
            vect = np.array(list(features_dict.values()))

            # filter the values for bf prediction
            vect = vect[self.clf.mask]

            classification = types[self.clf.predict(vect)[0]]
            prob = self.clf.predictProb(vect)

            start_time = float(windowCount * self.adjustedWindow)/float(self.sampleRate)
            end_time = float((windowCount+1) * self.adjustedWindow)/float(self.sampleRate)

            windowCount +=1

            processed.append({'type': classification, 'probabilities': prob, 'start': start_time,
                                'end': end_time, 'feats': features_dict, 'count': 1})
        return processed

    # test method
    def marginSmoothing(self, processed):
        smoothingDepth = 3
        numSegments = len(processed)
        if processed[0]['type'] == 'fore':
            labels={'fore':0, 'back':0, 'backfore':0}
            # get average of classes in the smoothing depth
            for i in range(1, min(smoothingDepth+1, numSegments)):
                categ = processed[i]['type']
                labels[categ] += 1
            # assign the most common type within smoothing depth to the beginning
            processed[0]['type'] = max(labels, key=labels.get)

        if processed[-1]['type'] == 'fore':
            labels={'fore':0, 'back':0, 'backfore':0}
            # get average of classes in the smoothing depth
            for i in range(max(0, numSegments-smoothingDepth-1), numSegments-1):
                print('i = ', i)
                categ = processed[i]['type']
                labels[categ] += 1
            # assign the most common type within smoothing depth to the beginning
            processed[-1]['type'] = max(labels, key=labels.get)
            print(labels)

        return processed


    # anterior foreground expansion, reclassify segments before fg segments as fg if they fall under a certain probability difference
    # aims to include the beginning of fg sounds in the fg cluster
    def foregroundExpansion(self, processed, k_depth = 3):
        for index in range(0,len(processed)):
            # If we have a fg
            if processed[index]['type'] == 'fore':
                print('we have detected a fg at index %d'% index)
                # check the previous
                prev = index - 1
                if prev >= 0:
                    prev_probs = processed[prev]['probabilities'][0]
                    print('previous probabilities are: ', prev_probs)
                    difference = max(prev_probs) - prev_probs[0]
                    print('difference is ', difference)
                    if difference < 1:
                        print('CHANGING index %d TO FG'% prev)
                        processed[prev]['type'] = 'fore'
        return processed

    # K Means clustering - renaming segments giving preference to foreground (default val of 3)
    def foregroundClustering(self, processed, k_depth = 3):
        start = 0
        while start < len(processed):
            # If we have a fg
            if processed[start]['type'] == 'fore':
                log_a = log_b =start
                # Go through k deep and save the idx of furthest fg within k
                for i in range(start+1, start+k_depth+1, 1):
                    if i < len(processed):
                        categ = processed[i]['type']
                        if categ == 'fore':
                            log_b = i
                # now we overwrite the types between the two detected foregrounds if we found one
                if log_b - log_a > 0:
                    for j in range(log_a, log_b+1,1): 
                            processed[j]['type'] = 'fore'
                    start = log_b
                # we didnt find a fg withing the k window
                # continue and skip remeinder of the window since theres no fg within it
                else: 
                    start += k_depth
            # not fg, move to next element
            else:
                start += 1
        return processed

    def medianFiltering(self, segments):
        if(self.filterWindow == 0):
            return segments
        import operator
        filtered = []
        for i in range(0,len(segments),1):
            # if segments[i]['type'] == 'fore':
            if (segments[i]['type'] == 'fore'):
                if(i > 1 and i < len(segments) - 2):
                    filtered.append('fore')
                    continue
            labels={'fore':0, 'back':0, 'backfore':0}
            for j in range(max(0,i-self.filterWindow), min(i+self.filterWindow,len(segments)), 1):
                k = segments[j]['type']
                labels[k] = labels[k] + 1
            maxlabel = max(labels.items(), key=operator.itemgetter(1))[0]
            filtered.append(maxlabel)

        for i in range(0,len(segments),1):
            if segments[i]['type'] != filtered[i]:
               print (i,'change',segments[i]['type'])
               segments[i]['type'] = filtered[i]
               print (' to ',segments[i]['type'])

        return segments

    # a simple median filtering
    def max_posterior(self, processed, m_span):
        import operator
        medWin = m_span#int(floor(m_span/2))
        filtered = []
        for i in range(0,len(processed),1):
            #print(i,'old',processed[i]['type'])
            #if processed[i]['type'] != processed[i+1]['type']:
            labels={'back':0, 'fore':0, 'backfore':0}
            for j in range(max(0,i-medWin), min(i+medWin,len(processed)), 1):
                k = processed[j]['type']
                labels[k] = labels.setdefault(k, 0) + 1
            maxlabel = max(labels.items(), key=operator.itemgetter(1))[0]
            filtered.append(maxlabel)

        for i in range(0,len(processed),1):
            #print (i,'old',processed[i]['type'])
            if processed[i]['type'] != filtered[i]:
               processed[i]['type'] = filtered[i]
               #print (i,'change',processed[i]['type'])

        return processed

    def smoothProbabilities(self, processed, winSize=200):
        a = np.array(processed[0]['probabilities'])
        
        for i in range(0,len(processed),1):
            #print processed[i]['probabilities']
            a = np.vstack((a,processed[i]['probabilities']))
        #print a
        
        for i in range(a.shape[1]):
            a[:,i]= ndimage.filters.median_filter(a[:, i], size=winSize)
        #print a

        import operator
        labels = ['fore', 'back', 'backfore']
        for i in range(0,len(processed),1):
            processed[i]['probabilities'] = a[i]
            index, value = max(enumerate(a[i]), key=operator.itemgetter(1))
            processed[i]['type']=labels[index]
        return processed

    

        # Here we join up any same labelled adjacent regions
    def conjunction(self, processed):
        for i in range(1, len(processed), 1):
            if processed[i]['type'] == processed[i-1]['type']:  # if its the same
                processed[i]['start'] = processed[i - 1]['start']  # update the start time
                processed[i]['feats'] = self.sumFeatureDics(
                    processed[i]['feats'], processed[i-1]['feats'])
                processed[i]['count'] += processed[i-1]['count']
                processed[i-1]['type'] = 'none'  # nullify the previos segment
            else:
                pass
            
        print('Finished conjunction')
        return self.finalize_regions(processed)

    def finalize_regions(self, processed):
        region_data = []
        for i in processed:
            if i['type'] != 'none':
                temp = {}
                temp['type'] = i['type']
                temp['duration'] = i['end'] - i['start']  # duration
                temp['start'] = i['start']
                temp['end'] = i['end']
                temp['feats'] = self.avgDicItems(i['feats'], i['count'])
                # unpack features and apply masks for valence and arousal
                f = temp['feats']
                vect = np.array(list(f.values()))
                arousal_vect = vect[self.afp.arousal_mask]
                valence_vect = vect[self.afp.valence_mask]
                temp['arousal'] = self.afp.predict_arousal(arousal_vect)
                temp['valence'] = self.afp.predict_valence(valence_vect)

                temp['probabilities'] = i['probabilities'] #remove after testign

                region_data.append(temp)
        return region_data

    def avgDicItems(self, D, a):
        result = {}
        for key in D.keys():
            D[key]/a
            result[key] = D[key]/a
        return result

    def sumFeatureDics(self, Da, Db):
        result = {}
        for key in Da.keys():
            A = Da[key]
            B = Db[key]
            result[key] = A+B  # [a + b for (a,b) in zip(A,B)]
        return result

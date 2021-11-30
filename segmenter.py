from math import sqrt, floor
from essentia_engine import EssentiaEngine
import bf_classifier
import affect_predictor
import numpy as np
from scipy import ndimage

from essentia.standard import MonoLoader, FrameGenerator, PoolAggregator
import essentia.utils as utils
import essentia

# train model
# segment new files

debug = False

#TODO
# smooth probabilities?
# processfile?

class Segmenter:
    # initialize
    def __init__(self):

        # create the models
        self.clf = bf_classifier.BFClassifier()
        self.afp = affect_predictor.AffectPredict()

        self.windowDuration = 0.5 # analysis window length in seconds
        self.sampleRate = 44100  # sample rate
        self.frameSize = 2048  # samples in each frame
        self.hopSize = 1024
        self.windowSize = int(self.sampleRate * self.windowDuration)

        # the essentia engine make sure that the features were extracted under the same conditions as the training data
        self.engine = EssentiaEngine(self.sampleRate, self.frameSize, self.hopSize)

    # run the segmentation
    def segment(self, afile):
        rawRegions = self.extractRegions(afile)
        # for item in rawRegions:
        #     print(item)
        clusteredRegions = self.Clustering(rawRegions)
        finalRegions = self.conjunction(clusteredRegions)
        return finalRegions

    # audio file path
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

        # translate type naming convention from csv to database 
        types = {'background': 'back', 'foreground': 'fore', 'bafoground': 'backfore'}

        processed = []  # storage for the classified segments

        for window in FrameGenerator(audio, frameSize=self.windowSize, hopSize=self.windowSize, startFromZero=True, lastFrameToEndOfFile=True):
            # extract all features
            pool = self.engine.extractor(window)
            aggrigatedPool = essentia.standard.PoolAggregator(defaultStats=['mean', 'stdev', 'skew', 'dmean', 'dvar', 'dmean2', 'dvar2'])(pool)

            # compute mean and variance of the frames using the pool aggregator, assign to dict in same order as training
            # narrow everything down to select features
            features_dict = {}
            descriptorNames = aggrigatedPool.descriptorNames()

            values = []
            descriptorList = []

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
            vect = vect[self.clf.features]

            classification = types[self.clf.predict(vect)[0]]
            prob = self.clf.predictProb(vect)

            start_time = float(windowCount * self.windowSize)/float(self.sampleRate)
            end_time = float((windowCount+1) * self.windowSize)/float(self.sampleRate)

            windowCount +=1

            processed.append({'type': classification, 'probabilities': prob, 'start': start_time,
                                'end': end_time, 'feats': features_dict, 'count': 1})
        return processed

    # K Means clustering - renaming segments giving preference to foreground (default val of 3)
    def Clustering(self, processed, k_depth = 3):
        start = 0
        while start < len(processed):
            # If we have a fg
            if processed[start]['type'] == 'foreground':
                log_a = log_b =start
                # Go through k deep and save the idx of furthest fg within k
                for i in range(start+1, start+k_depth+1, 1):
                    if i < len(processed):
                        categ = processed[i]['type']
                        if categ == 'foreground':
                            log_b = i
                # now we overwrite the types between the two detected foregrounds if we found one
                if log_b - log_a > 0:
                    for j in range(log_a, log_b+1,1): 
                            processed[j]['type'] = 'foreground'
                    start = log_b
                # we didnt find a fg withing the k window
                # continue and skip remeinder of the window since theres no fg within it
                else: 
                    start += k_depth
            # not fg, move to next element
            else:
                start += 1
        return processed

    def conjunction(self, processed):
        # Here we join up any same labelled adjacent regions
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

from math import sqrt, floor
from essentia_engine import EssentiaEngine
import bf_classifier
# import affect_predictor
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
        # self.afp = affect_predictor.AffectPredict()TODO

        self.windowDuration = 0.5 # analysis window length in seconds
        self.sampleRate = 44100  # sample rate
        self.frameSize = 1024  # samples in each frame
        self.hopSize = 512

        # the essentia engine make sure that the features were extracted under the same conditions as the training data
        self.engine = EssentiaEngine(self.sampleRate, self.frameSize)

    # run the segmentation
    def segment(self, afile):
        rawRegions = self.extractRegions(afile)
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
        frameCount_window = 0
        frameCount_file = 0
        windowCount = 0

        # calculate the length of analysis frames
        frame_duration = float(self.frameSize / 2)/float(self.sampleRate)
        # number frames in a window
        numFrames_window = int(self.windowDuration / frame_duration)

        print(numFrames_window, ' frames in a window')
        print('frame duration: ', frame_duration)
        print('audio len: ', len(audio))
        print('number frames total: ', len(audio)/self.frameSize)

        # dictionary for class names from libsvm format
        types = {1: 'back', 2: 'fore', 3: 'backfore'}

        processed = []  # storage for the classified segments

        for frame in FrameGenerator(audio, frameSize=self.frameSize, hopSize=self.hopSize, startFromZero=True, lastFrameToEndOfFile=True):

            # spectral contrast valleys
            frame_windowed = self.engine.get_window(frame)
            frame_spectrum = self.engine.get_spectrum(frame_windowed)
            sc_valley = self.engine.get_spectral_contrast(frame_spectrum)
            pool.add('lowlevel.spectral_contrast_valleys', sc_valley)
            # silence rate
            pool.add('lowlevel.silence_rate',
                     self.engine.get_silence_rate(frame))
            # spectral flux
            pool.add('lowlevel.spectral_flux',
                     self.engine.get_spectral_flux(frame_spectrum))
            # Gammatone-frequency cepstral coefficients
            gfccs = self.engine.get_gfcc(frame_spectrum)
            pool.add('lowlevel.gfcc', gfccs)
            # spectral RMS
            pool.add('lowlevel.spectral_rms',
                     self.engine.get_rms(frame_spectrum))
            # increment counters
            frameCount_window += 1
            frameCount_file += 1

            # detect if we have traversed a whole window
            if (frameCount_window == numFrames_window):
                windowCount +=1

                # replay gain is not frame level like the others, get it for the whole window
                try:
                    window_start = int((windowCount - 1) * self.windowDuration * self.sampleRate)
                    window_end =  int(windowCount * self.windowDuration * self.sampleRate)
                    replay_gain = self.engine.get_rgain(audio[ window_start : window_end])
                    replay_gain_previous = replay_gain
                except:
                    # if window is too small to get replay gain, use the value from the previous window
                    print('window size to small for replay gain algorithm, using previous value')
                    replay_gain = replay_gain_previous


                # compute mean and variance of the frames using the pool aggregator, assign to dict in same order as training
                aggrPool = PoolAggregator(defaultStats=['mean', 'stdev'])(pool)
                features_dict = {}
                features_dict['lowlevel.silence_rate.stdev'] = aggrPool['lowlevel.silence_rate.stdev']
                features_dict['lowlevel.spectral_contrast_valleys.mean.0'] = aggrPool['lowlevel.spectral_contrast_valleys.mean'][0]
                features_dict['lowlevel.spectral_contrast_valleys.stdev.2'] = aggrPool['lowlevel.spectral_contrast_valleys.stdev'][2]
                features_dict['lowlevel.spectral_contrast_valleys.stdev.3'] = aggrPool['lowlevel.spectral_contrast_valleys.stdev'][3]
                features_dict['lowlevel.spectral_contrast_valleys.stdev.4'] = aggrPool['lowlevel.spectral_contrast_valleys.stdev'][4]
                features_dict['lowlevel.spectral_contrast_valleys.stdev.5'] = aggrPool['lowlevel.spectral_contrast_valleys.stdev'][5]
                features_dict['lowlevel.spectral_flux.mean'] = aggrPool['lowlevel.spectral_rms.mean']
                features_dict['lowlevel.gfcc.mean.0'] = aggrPool['lowlevel.gfcc.mean'][0]
                features_dict['lowlevel.spectral_rms.mean'] = aggrPool['lowlevel.spectral_rms.mean']
                features_dict['replay_gain'] = replay_gain

                # reset counter and clear pool
                frameCount_window = 0
                pool.clear()
                aggrPool.clear()

                # prepare feature values to predict the class
                vect = list(features_dict.values())
                type = self.clf.predict(vect)[0]
                prob = self.clf.predictProb(vect)
                # print(vect)
                start_time = float((frameCount_file - numFrames_window))*(self.frameSize/2)/float(self.sampleRate)
                end_time = float(frameCount_file*(self.frameSize/2))/float(self.sampleRate)
                processed.append({'type': type, 'probabilities': prob, 'start': start_time,
                                 'end': end_time, 'feats': features_dict, 'count': 1})
                # print('\n', processed[-1])
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
        if debug:
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
                f = temp['feats']
                vect = [f['lowlevel.silence_rate.stdev'], f['lowlevel.spectral_contrast_valleys.mean.0'], f['replay_gain'], f['lowlevel.spectral_contrast_valleys.stdev.2'],
                        f['lowlevel.spectral_contrast_valleys.stdev.3'], f['lowlevel.spectral_contrast_valleys.stdev.4'], f['lowlevel.spectral_contrast_valleys.stdev.5'], f['lowlevel.spectral_flux.mean'], f['lowlevel.gfcc.mean.0'], f['lowlevel.spectral_rms.mean']]
                # temp['valence'] = self.afp.predict_valence(vect) TODO
                # temp['arousal'] = self.afp.predict_arousal(vect) TODO
                region_data.append(temp)
        return region_data

    def avgDicItems(self, D, a):
        result = {}
        for key in D.keys():
            result[key] = D[key]/a
        return result

    def sumFeatureDics(self, Da, Db):
        result = {}
        for key in Da.keys():
            A = Da[key]
            B = Db[key]
            result[key] = A+B  # [a + b for (a,b) in zip(A,B)]
        return result

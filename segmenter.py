from essentia_engine import EssentiaEngine
import bf_classifier
import bf_ridge
import affect_predictor
import numpy as np
from scipy import ndimage
from essentia.standard import MonoLoader, FrameGenerator, PoolAggregator
import essentia

class Segmenter:
    # initialize
    def __init__(self):

        # create the models
        # self.clf = bf_classifier.BFClassifier()
        self.clf = bf_ridge.BFRidge()
        self.afp = affect_predictor.AffectPredict()

        self.window_duration = 1.5 # analysis window length in seconds
        self.sample_rate = 22050  # sample rate
        self.frame_size = 2048  # samples in each frame
        self.hop_size = 1024
        self.window_size = int(self.sample_rate * self.window_duration)
        self.adjusted_window = (self.window_size // self.frame_size) * self.frame_size

        self.smoothing_window = 1
        self.median_filter_span = 3

        self.filter_window = 3

        # the essentia engine make sure that the features were extracted under the same conditions as the training data
        self.engine = EssentiaEngine(
            self.sample_rate, self.frame_size, self.hop_size)

    # run the segmentation
    def segment(self, afile):
        segments = self.extract_regions(afile)

        # segment filtering
        segments = self.margin_smoothing(segments)

        # segments = self.smooth_probabilities(segments, self.smoothing_window)
        # segments = self.max_posterior(segments, self.medianFilter_span)
        # segments = self.median_filtering(segments)

        segments = self.kmeans_clustering(segments, 2, 'fore')
        segments = self.kmeans_clustering(segments, 2, 'backfore')
        # segments = self.kmeans_clustering(segments, 1, 'back')

        # join segments
        segments = self.conjunction(segments)
        return segments 

    # use the bf classifier to extract background, foreground, bafoground regions
    # returns # [file_path, [['type', start, end], [...], ['type'n, startn, endn]]]
    def extract_regions(self, afile):

        # instantiate the loading algorithm
        loader = MonoLoader(filename=afile, sampleRate=self.sample_rate)
        # perform the loading
        audio = loader()

        # create pool for storage and aggregation
        pool = essentia.Pool()

        # frame counter used to detect end of window
        windowCount = 0

        # calculate the length of analysis frames
        frame_duration = float(self.frame_size / 2)/float(self.sample_rate)

        # number frames in a window
        numFrames_window = int(self.window_duration / frame_duration)

        print(numFrames_window, ' frames in a window')
        print('frame duration: ', frame_duration)
        print('audio len: ', len(audio))
        print('number frames total: ', len(audio)/self.frame_size)
        print('window size: ', self.window_size)
        print('frame adjusted window size: ', self.adjusted_window)

        # translate type naming convention from csv to database
        types = {1: 'fore', 2: 'back', 3: 'backfore'}

        processed = []  # storage for the classified segments

        for window in FrameGenerator(audio, frameSize=self.adjusted_window, hopSize=self.adjusted_window, startFromZero=True, lastFrameToEndOfFile=True):
            # extract all features
            pool = self.engine.extractor(window)
            aggrigated_pool = PoolAggregator(defaultStats=['mean', 'stdev', 'skew', 'dmean', 'dvar', 'dmean2', 'dvar2'])(pool)

            # compute mean and variance of the frames using the pool aggregator, assign to dict in same order as training
            # narrow everything down to select features
            features_dict = {}
            descriptor_names = aggrigated_pool.descriptorNames()

            # unpack features in lists
            for descriptor in descriptor_names:
                # little to no values in these features, ignore
                if('tonal' in descriptor or 'rhythm' in descriptor):
                    continue
                value = aggrigated_pool[descriptor]
                # unpack arrays
                if (str(type(value)) == "<class 'numpy.ndarray'>"):
                    for idx, subVal in enumerate(value):
                        features_dict[descriptor + '.' + str(idx)] = subVal
                    continue
                # ignore strings
                elif(isinstance(value, str)):
                    pass
                # add singular values
                else:
                    features_dict[descriptor] = value

            # reset counter and clear pool
            pool.clear()
            aggrigated_pool.clear()

            # prepare dictionary for filtering
            vector = np.array(list(features_dict.values()))
            fnames = np.array(list(features_dict.keys()))

            # filter the features for bf prediction
            vect_filtered = vector

            # filter the feature dictionary to store only select features
            fnames_filtered = fnames

            # create filtered dictionary for the database
            features_filtered = {}
            for idx, val in enumerate(vect_filtered):
                features_filtered[fnames_filtered[idx]] = val

            # get the classification
            classification = types[self.clf.predict(vect_filtered)[0]]

            # get probabilities
            probabilities = self.clf.predict_prob(vect_filtered)

            start_time = float(windowCount * self.adjusted_window)/float(self.sample_rate)
            end_time = float((windowCount+1) * self.adjusted_window)/float(self.sample_rate)

            windowCount += 1

            processed.append({'type': classification, 'start': start_time,
                             'end': end_time, 'feats_select': features_filtered, 'vector': vector, 'count': 1, 'probabilities':probabilities})
        return processed

    # test method
    def margin_smoothing(self, processed):
        smoothing_depth = 2
        num_segments = len(processed)
        if processed[0]['type'] == 'fore':
            labels = {'fore': 0, 'back': 0, 'backfore': 0}
            # get average of classes in the smoothing depth
            for i in range(1, min(smoothing_depth+1, num_segments)):
                categ = processed[i]['type']
                labels[categ] += 1
            # assign the most common type within smoothing depth to the beginning
            processed[0]['type'] = max(labels, key=labels.get)

        if processed[-1]['type'] == 'fore':
            labels = {'fore': 0, 'back': 0, 'backfore': 0}
            # get average of classes in the smoothing depth
            for i in range(max(0, num_segments-smoothing_depth-1), num_segments-1):
                print('i = ', i)
                categ = processed[i]['type']
                labels[categ] += 1
            # assign the most common type within smoothing depth to the beginning
            processed[-1]['type'] = max(labels, key=labels.get)

        return processed

    # K Means clustering - renaming segments giving preference to foreground (default val of 3)
    def kmeans_clustering(self, processed, k_depth, category):
        start = 0
        while start < len(processed):
            if processed[start]['type'] == category:
                log_a = log_b = start
                # Go through k deep and save the idx of furthest fg within k
                for i in range(start+1, start+k_depth+2, 1):
                    if i < len(processed):
                        categ = processed[i]['type']
                        if categ == category:
                            log_b = i
                # now we overwrite the types between the two detected foregrounds if we found one
                if log_b - log_a > 1:
                    for j in range(log_a+1, log_b+1, 1):
                        processed[j]['type'] = category
                    start = log_b
                # we didnt find a fg withing the k window
                # continue and skip remeinder of the window since theres no fg within it
                else:
                    start += k_depth
            else:
                start += 1
        return processed

    def median_filtering(self, segments):
        if(self.filter_window == 0):
            return segments
        import operator
        filtered = []
        for i in range(0, len(segments), 1):
            if (segments[i]['type'] == 'fore'):
                filtered.append('fore')
                continue
            labels = {'fore': 0, 'back': 0, 'backfore': 0}
            for j in range(max(0, i-self.filter_window), min(i+self.filter_window, len(segments)), 1):
                k = segments[j]['type']
                labels[k] = labels[k] + 1
            maxlabel = max(labels.items(), key=operator.itemgetter(1))[0]
            filtered.append(maxlabel)

        for i in range(0, len(segments), 1):
            if segments[i]['type'] != filtered[i]:
                print(i, 'change', segments[i]['type'])
                segments[i]['type'] = filtered[i]
                print(' to ', segments[i]['type'])

        return segments

    # a simple median filtering
    def max_posterior(self, processed, m_span):
        import operator
        medWin = m_span  # int(floor(m_span/2))
        filtered = []
        for i in range(0, len(processed), 1):
            # print(i,'old',processed[i]['type'])
            # if processed[i]['type'] != processed[i+1]['type']:
            labels = {'back': 0, 'fore': 0, 'backfore': 0}
            for j in range(max(0, i-medWin), min(i+medWin, len(processed)), 1):
                k = processed[j]['type']
                labels[k] = labels.setdefault(k, 0) + 1
            maxlabel = max(labels.items(), key=operator.itemgetter(1))[0]
            filtered.append(maxlabel)

        for i in range(0, len(processed), 1):
            #print (i,'old',processed[i]['type'])
            if processed[i]['type'] != filtered[i]:
                processed[i]['type'] = filtered[i]
                #print (i,'change',processed[i]['type'])

        return processed

    def smooth_probabilities(self, processed, winSize=200):
        a = np.array(processed[0]['probabilities'])

        for i in range(0, len(processed), 1):
            # print processed[i]['probabilities']
            a = np.vstack((a, processed[i]['probabilities']))
        # print a

        for i in range(a.shape[1]):
            a[:, i] = ndimage.filters.median_filter(a[:, i], size=winSize)
        # print a

        import operator
        labels = ['fore', 'back', 'backfore']
        for i in range(0, len(processed), 1):
            processed[i]['probabilities'] = a[i]
            index, value = max(enumerate(a[i]), key=operator.itemgetter(1))
            processed[i]['type'] = labels[index]
        return processed

        # Here we join up any same labelled adjacent regions

    def conjunction(self, processed):
        for i in range(1, len(processed), 1):
            if processed[i]['type'] == processed[i-1]['type']:  # if its the same
                processed[i]['start'] = processed[i - 1]['start']  # update the start time
                processed[i]['feats_select'] = self.sum_feature_dicts(
                    processed[i]['feats_select'], processed[i-1]['feats_select'])
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
                temp['feats_select'] = self.avg_dict_items(i['feats_select'], i['count'])

                # unpack features and apply masks for valence and arousal
                arousal_vect = i['vector'][self.afp.AROUSAL_MASK]
                valence_vect = i['vector'][self.afp.VALENCE_MASK]
                temp['arousal'] = self.afp.predict_arousal(arousal_vect)
                temp['valence'] = self.afp.predict_valence(valence_vect)

                temp['probabilities'] = i['probabilities']

                region_data.append(temp)
        return region_data

    def avg_dict_items(self, D, a):
        result = {}
        for key in D.keys():
            D[key]/a
            result[key] = D[key]/a
        return result

    def sum_feature_dicts(self, Da, Db):
        result = {}
        for key in Da.keys():
            A = Da[key]
            B = Db[key]
            result[key] = A+B  # [a + b for (a,b) in zip(A,B)]
        return result

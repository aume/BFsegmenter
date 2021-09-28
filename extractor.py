# first, we need to import our essentia module. It is aptly named 'essentia'!
import essentia
import essentia.standard
import numpy as np

excludedMetrics = ['rhythm.beats_position', 'rhythm.bpm_histogram', 'lowlevel.mfcc.icov', 'lowlevel.mfcc.cov', 'lowlevel.gfcc.icov', 'lowlevel.gfcc.cov']

def getEverything(file):
    # Compute all features, aggregate only 'mean' and 'stdev' statistics for all low-level, rhythm and tonal frame features
    features, features_frames = essentia.standard.MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                                                 rhythmStats=[
                                                                     'mean', 'stdev'],
                                                                 tonalStats=['mean', 'stdev'])(file)
    
    #, lowlevelFrameSize = 1024, lowlevelHopSize = 512, analysisSampleRate = 44100

    descriptorNames = features.descriptorNames()

    descriptorList = []
    values = []
    
    for descriptor in descriptorNames:
        value = features[descriptor]
        # exclude certain metrics for now (list of lists)
        if(descriptor in excludedMetrics):
            continue

        if (str(type(value)) == "<class 'numpy.ndarray'>"):
            for idx, subVal in enumerate(value):
                descriptorList.append(descriptor + '.' + str(idx))
                values.append(subVal)
            continue

        elif(type(value) is str):
            continue # NOTE: possibly add some of these later

        else:
            descriptorList.append(descriptor)
            values.append(repr(value))

    # print(features_frames['lowlevel.spectral_contrast_valleys'])
    sc_coeffs0 = np.mean([x[0] for x in features_frames['lowlevel.spectral_contrast_valleys']])
    print(sc_coeffs0)
    
    return descriptorList, values
from processing import partitionSelectData
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class BFClassifier(object):
    """
        The famous classier for segmenting audio files into background, 
        foreground, and background with foreground segments
    """

    def __init__(self):
        super().__init__()
        
        # get the training data
        self.features = [44, 69, 95, 188, 195, 430, 472, 500, 531, 536, 539, 542, 549, 561, 568, 577, 580]
        self.feature_names = ['lowLevel.pitch_instantaneous_confidence.dmean2', 'lowLevel.silence_rate_30dB.skew', 'lowLevel.spectral_crest.dvar2', 'lowLevel.spectral_spread.skew', 'lowLevel.spectral_strongpeak.skew', 'lowLevel.mfcc.dmean2.6', 'lowLevel.mfcc.mean.9', 'lowLevel.mfcc.stdev.11', 'lowLevel.sccoeffs.mean.5', 'lowLevel.sccoeffs.skew.4', 'lowLevel.sccoeffs.stdev.1', 'lowLevel.sccoeffs.stdev.4', 'lowLevel.scvalleys.dmean.5', 'lowLevel.scvalleys.dvar.5', 'lowLevel.scvalleys.mean.0', 'lowLevel.scvalleys.skew.3', 'lowLevel.scvalleys.stdev.0']
        featureVectors, classList, featureNames = partitionSelectData('datasets/features_BF200.csv', self.features)
        print('first vector: ',featureVectors[0])
        # create model, scale the data using a pipeline 
        # computes the mean and standard deviation on the training set so as to be able to later re-apply the same transformation on the testing set
        self.pipe = make_pipeline(StandardScaler(), svm.SVC(kernel = 'rbf', C=1, probability = True))
        # train the model
        self.pipe.fit(featureVectors, classList) 
        print('model fit')
    
    def predict(self, features):
        return self.pipe.predict([features])

    def predictProb(self, features):
        return self.pipe.predict_log_proba([features])

    
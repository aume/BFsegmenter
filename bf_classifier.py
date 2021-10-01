from processing import selectFeaturesToLists, featuresToLists
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
        featureVectors, classList = featuresToLists('datasets\manualfeatures_BF200.csv')

        # create model, scale the data using a pipeline 
        # computes the mean and standard deviation on the training set so as to be able to later re-apply the same transformation on the testing set
        self.pipe = make_pipeline(StandardScaler(), svm.SVC(kernel = 'rbf', probability = True, cache_size = 1000))
        # train the model
        self.pipe.fit(featureVectors, classList) 
    
    def predict(self, features):
        return int(self.pipe.predict(features))

    def predictProb(self, features):
        return self.pipe.predict_log_proba(features)

    
from processing import selectFeaturesToLists
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class BFClassifier(object):
    """
        The famous classier for segmenting audio files into background, 
        foreground, and background with foreground segments
    """

    def __init__(self):
        super(BFClassifier, self).__init__()
        
        # select desired features, create training data
        CorrelationAttributeEval10 = [50, 629, 93, 638, 69, 637, 255, 73, 640, 639]
        featureVectors, classList = selectFeaturesToLists(CorrelationAttributeEval10, 'extractedfeatures/features_BF200.csv')

        # create model, scale the data using a pipeline 
        # computes the mean and standard deviation on the training set so as to be able to later re-apply the same transformation on the testing set
        self.pipe = make_pipeline(StandardScaler(), svm.SVC(kernel = 'rbf', probability = True, cache_size = 1000))
        # train the model
        self.pipe.fit(featureVectors, classList) 
    
    # TODO need to fix this?
    def predict(self, features):
        return int(self.pipe.predict(features)[0])

    # TODO need to fix this?
    def predictProb(self, features):
        return self.pipe.predict_log_proba(features)[0]

    
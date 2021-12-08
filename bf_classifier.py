from processing import partitionSelectData
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

class BFClassifier(object):
    """
        The famous classier for segmenting audio files into background, 
        foreground, and background with foreground segments
    """

    def __init__(self):
        print('initializing classifier')
        super().__init__()

        fa = open('datasets/features_BF200.csv','r')
        self.header = fa.readline().split(',')
        bf_data = np.loadtxt(fa,delimiter=",")

        train_y = bf_data[:,-1:]
        train_X = bf_data[:,0:-1]

        # verify correct training data length
        assert len(train_y) == len(train_X)

        # masks to select features
        self.mask = [49, 71, 77, 88, 95, 104, 125, 144, 153, 173, 216, 247, 255, 482, 561, 568, 580]

        # apply mask to get select features only
        train_X = [x[self.mask] for x in train_X]

        # create model, scale the data using a pipeline 
        # computes the mean and standard deviation on the training set so as to be able to later re-apply the same transformation on the testing set
        self.pipe = make_pipeline(StandardScaler(), svm.SVC(C=0.12742749857031335, kernel='linear', probability=True))
        # train the model
        self.pipe.fit(train_X, train_y.ravel()) 
        print('model fit succesfully')
    
    def predict(self, features):
        return self.pipe.predict([features])

    def predictProb(self, features):
        return self.pipe.predict_log_proba([features])

    
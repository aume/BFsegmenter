#!/usr/bin/env python

from sklearn.datasets import load_svmlight_file
from sklearn import svm

class BFClassifier(object):
    """
        The famous classier for segmenting audio files into background, 
        foreground, and background with foreground segments
    """
    def __init__(self):
        super(BFClassifier, self).__init__()
        
        # load the svm format training data
        #<label> <feature-id>:<feature-value> <feature-id>:<feature-value>
        x_train, y_train = load_svmlight_file('./feature_output.txt')
        # create the model
        self.clf = svm.SVC(kernel='poly', probability=True)
        #train the model
        self.clf.fit(x_train, y_train)
        
    def predict(self, features):
        return int(self.clf.predict(features)[0])
        
    def predictProb(self, features):
        return self.clf.predict_log_proba(features)[0]
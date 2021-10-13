import numpy as np
from sklearn import linear_model

f = open('datasets/valencearousal_training.csv','r')

header = f.readline().split(',')

training_data = np.loadtxt(f)

arousal_data = training_data[:,9:10] # first two colums are valence and arousal
valence_data = training_data[:,10:11] # first two colums are valence and arousal
features_data = training_data[:,0:9]

# verify correct training data length
assert len(arousal_data) == len(features_data)
assert len(valence_data) == len(features_data)

valence_model = linear_model.LinearRegression()
valence_model.fit(features_data, valence_data)

arousal_model = linear_model.LinearRegression()
arousal_model.fit(features_data, arousal_data)
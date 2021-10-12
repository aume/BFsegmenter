import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

f = open('olddata/studyData_.csv','r')

header = f.readline().split(',')

training_data = np.loadtxt(f,delimiter=",",skiprows=1)

Y = training_data[:,0:2] # first two colums are valence and arousal    
X = training_data[:,2:len(training_data)] # remaining columns are audio features

assert len(Y) == len(X)

valence_model = linear_model.LinearRegression() # Ridge (alpha = .5)
valence_model.fit(X,Y[:,0])

print(Y)

arousal_model = linear_model.LinearRegression()
arousal_model.fit(X,Y[:,1])
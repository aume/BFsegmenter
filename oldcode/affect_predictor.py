
#!/usr/bin/env python

import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

class AffectPredict:
    """docstring for AffectPredict"""
    
    def __init__(self):
        
        f = open('studyData_.csv','rb')
        
        self.header = f.readline().split(',')
        training_data = np.loadtxt(f,delimiter=",",skiprows=1)
        
        self.Y = training_data[:,0:2] # first two colums are valence and arousal    
        self.X = training_data[:,2:len(training_data)] # remaining columns are audio features
        
        assert len(self.Y) == len(self.X)
        
        self.valence_model = linear_model.LinearRegression() # Ridge (alpha = .5)
        self.valence_model.fit(self.X,self.Y[:,0])
        
        self.arousal_model = linear_model.LinearRegression()
        self.arousal_model.fit(self.X,self.Y[:,1])
        
        
    def predict_valence(self, Z):
        
        return self.valence_model.predict(Z)
        
    def predict_arousal(self, Z):
       
        return self.arousal_model.predict(Z)
    
    def model_stats(self):

        print("Valence RSS: %.2f"
            % np.mean((self.valence_model.predict(self.X) - self.Y[:,0]) ** 2))
        print("Arousal RSS: %.2f"
            % np.mean((self.arousal_model.predict(self.X) - self.Y[:,1]) ** 2))
        print('Valence variance score: %.2f' % self.valence_model.score(self.X, self.Y[:,0]))
        print('Arousal variance score: %.2f' % self.arousal_model.score(self.X, self.Y[:,1]))
            
    def visualize_model(self, x, y, m, c):
        '''
            TODO: needs finxin
        '''
        w = 2
        h = 2#math.floor(len(y)/2)
        f, axarr = plt.subplots(w,h)
        count = 0
        for  i in range(w):
            for j in range(h):
                 axarr[i,j].plot(x[:,count], y, 'o', markersize=3) # label='Original data'
                 axarr[i,j].plot(x[:,count], m[count]*x[:,count]+c, 'r') # label='Fitted line'
                 axarr[i,j].set_title(self.header[2+count]) 
                 #axarr[i,j].set_yscale('exp')
                 count += 1  

        plt.legend()
        plt.show()
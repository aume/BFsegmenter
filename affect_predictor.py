import numpy as np
from sklearn.ensemble import RandomForestRegressor

class AffectPredict:
    '''
        Emotion prediction models for valence and arousal.
    '''
    def __init__(self):
        # masks to select features
        self.AROUSAL_MASK = [1, 14, 29, 30, 33, 35, 41, 47, 84, 87, 89, 90, 91, 93, 127, 128, 131, 132, 138, 140, 144, 166, 184, 191, 195, 199, 207, 216, 231, 236, 243, 245, 258, 261, 262, 263, 342, 343, 345, 351, 361, 373, 379, 393, 395, 398, 433, 437, 453, 463, 482, 486, 489, 502, 504, 517, 522, 526, 527, 528, 529, 540, 549, 553, 558, 561, 565, 572, 573, 585, 589, 594]
        self.VALENCE_MASK = [12, 24, 29, 30, 33, 45, 49, 85, 86, 89, 94, 106, 107, 118, 128, 134, 135, 140, 152, 159, 166, 185, 188, 195, 225, 232, 253, 254, 255, 259, 262, 349, 360, 374, 377, 380, 382, 387, 388, 389, 424, 437, 438, 445, 459, 466, 467, 470, 472, 480, 494, 498, 520, 531, 533, 535, 536, 537, 540, 555, 561, 567, 575, 579, 589, 593, 604]
        
        fa = open('datasets/arousal_data.csv','r')
        fv = open('datasets/valence_data.csv','r')

        self.arousal_header = fa.readline().split(',')
        self.valence_header = fv.readline().split(',')

        arousal_data = np.loadtxt(fa,delimiter=",")
        valence_data = np.loadtxt(fv,delimiter=",")

        self.arousal_y = arousal_data[:,-1:]
        self.arousal_X = arousal_data[:,0:-1]
        self.valence_y = valence_data[:,-1:]
        self.valence_X = valence_data[:,0:-1]

        # verify correct training data length
        assert len(self.arousal_X) == len(self.arousal_y)
        assert len(self.valence_X) == len(self.valence_y)

        # apply mask to get select features only
        self.arousal_X = [x[self.AROUSAL_MASK] for x in self.arousal_X]
        self.valence_X = [x[self.VALENCE_MASK] for x in self.valence_X]

        # create arousal model
        self.arousal_model = RandomForestRegressor(max_depth=40, min_samples_split=5, oob_score=True)
        self.arousal_model.fit(self.arousal_X, self.arousal_y.ravel())

        # create valence model
        self.valence_model = RandomForestRegressor(max_depth=40, oob_score=True)
        self.valence_model.fit(self.valence_X, self.valence_y.ravel())

    def predict_valence(self, Z):
        return self.valence_model.predict([Z]).item(0)

    def predict_arousal(self, Z):
        return self.arousal_model.predict([Z]).item(0)

    def model_stats(self):
        print('arousal r-squared score: %.2f' % self.arousal_model.oob_score_)
        print('valence r-squared score: %.2f' % self.valence_model.oob_score_)
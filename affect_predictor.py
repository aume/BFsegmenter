import numpy as np
from sklearn import linear_model

class AffectPredict:
    def __init__(self):
        f = open('datasets/valencearousal_training.csv','r')

        self.header = f.readline().split(',')
        training_data = np.loadtxt(f,delimiter=",")

        self.arousal_data = training_data[:,9:10] # first two colums are valence and arousal
        self.valence_data = training_data[:,10:11] # first two colums are valence and arousal
        self.features_data = training_data[:,0:10]

        # verify correct training data length
        assert len(self.arousal_data) == len(self.features_data)
        assert len(self.valence_data) == len(self.features_data)
        print('arousal len: ', len(self.features_data[0]))
        print(self.features_data[0])
        self.valence_model = linear_model.LinearRegression()
        self.valence_model.fit(self.features_data, self.valence_data)

        self.arousal_model = linear_model.LinearRegression()
        self.arousal_model.fit(self.features_data, self.arousal_data)

    def predict_valence(self, Z):
        return self.valence_model.predict([Z]).item(0)

    def predict_arousal(self, Z):
        return self.arousal_model.predict([Z]).item(0)

    def model_stats(self):
        print("Valence RSS: %.2f"
            % np.mean((self.valence_model.predict(self.features_data) - self.valence_data) ** 2))
        print("Arousal RSS: %.2f"
            % np.mean((self.arousal_model.predict(self.features_data) - self.arousal_data) ** 2))
        print('Valence variance score: %.2f' % self.valence_model.score(self.features_data, self.valence_data))
        print('Arousal variance score: %.2f' % self.arousal_model.score(self.features_data, self.arousal_data))


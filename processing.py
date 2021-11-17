import numpy as np
import csv

def selectFeaturesToLists(self, featureNumbers, featuresFilename):
    # add -1 to all features numbers to convert them to list indeces
    theChosenOnes = np.add(featureNumbers, -1).tolist()
    
    with open(featuresFilename, 'r') as inp:
        firstIteration = True
        featureVectors = []
        
        for row in csv.reader(inp):
            # if on first iteration, add the class index to the list of selected features
            if( firstIteration ):
                classIdx = len(row) - 1
                theChosenOnes.append(classIdx)
                firstIteration = False 

            # extract only the selected features and their corresponding classes
            featureVectors.append(list(np.array(row)[theChosenOnes]))

        # format data for model
        classList = []
        featureNames = featureVectors.pop(0) # remove first element (table titles)

        # remove the classes from the end of the list and append to their own
        for r in featureVectors:
            classList.append(r.pop())
    return featureVectors, classList

def featuresToLists(featuresFilename):

    with open(featuresFilename, 'r') as inp:
        firstIteration = True
        featureVectors = []
        
        for row in csv.reader(inp):
            # extract features and their corresponding classes
            featureVectors.append(list(row))
            

        # format data for model
        classList = []
        featureNames = featureVectors.pop(0) # remove first element (table titles)
        featureNames.pop(-1)

        # remove the classes from the end of the list and append to their own
        for r in featureVectors:
            classList.append(r.pop())
            
    return featureVectors, classList, featureNames

print('heeloo')
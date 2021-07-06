import os
import sys
from math import sqrt
''' 
argv[1] is the upper level directory path, argv[2] is the file sufix to search for
construct a list of dictionaries from file system
of file name, class, path

T
|
c1 --------------- c2 ... cn
|
mp3 - aiff - ogg
		|
	f1, f2, ..., fn 
use the dir name above file type as the class name
'''

# These are the features to be logged
featureNames = ['MFCC8_Mean','MFCC11_Mean', 'MFCC15_Mean', 'MFCC28_Mean', 'MFCC36_Mean',\
                'MFCC1_Std', 'MFCC2_Std', 'MFCC5_Std', 'MFCC6_Std', 'MFCC18_Std', 'MFCC20_Std', 'MFCC32_Std', 'MFCC34_Std',\
                'Loudness_Mean', 'Loudness_Std', 'PerceptualSpread_Mean', 'SpectralFlux_Std']

# FYI - ordering on the output
# 'Loudness_Mean', 'Loudness_Std', 'PerceptualSpread_Mean', 'SpectralFlux_Std', 'MFCC1_Std', 'MFCC2_Std', 'MFCC5_Std', 'MFCC6_Std', 'MFCC8_Mean', 'MFCC11_Mean', 'MFCC15_Mean', 'MFCC18_Std', 'MFCC20_Std', 'MFCC28_Mean', 'MFCC32_Std', 'MFCC34_Std', 'MFCC36_Mean'

diclist = []
for dirname, dirnames, filenames in os.walk(sys.argv[1]):
	print dirname
	for subdirname in dirnames:
		if subdirname == sys.argv[2]:
			validir = os.path.join(dirname, subdirname)
			for file in os.listdir(validir):
				if os.path.splitext(file) [1] == '.'+ sys.argv[2]:
					cname = os.path.split(os.path.dirname(validir))[1]
	     			#print (dirname, subdirname, file, cname)
					tdic = {'fname':file, 'class':cname, 'path':validir+'/'}
					diclist.append(tdic)
	
for dic in diclist:
	print (dic['fname'], dic['class'], dic['path'])

# The feature extraction setup

from yaafelib import *

blocksize = 1024
step = blocksize/2
srate = 22500

#FeaturePlan is a collection of features to extract, configured for a specific sample rate.
fp = FeaturePlan(sample_rate=srate, resample=True)

fp.addFeature('MFCC: MFCC CepsNbCoeffs=39 blockSize=%d stepSize=%d' % (blocksize, step))
fp.addFeature('Loudness: Loudness LMode=Total blockSize=%d stepSize=%d' % (blocksize, step))

#fp.addFeature('PerceptualSharpness: PerceptualSharpness blockSize=%d stepSize=%d' % (blocksize, step))
fp.addFeature('PerceptualSpread: PerceptualSpread blockSize=%d stepSize=%d' % (blocksize, step))
#fp.addFeature('PerceptualFlatnes: PerceptualFlatnes blockSize=%d stepSize=%d' % (blocksize, step))

fp.addFeature('SpectralFlux: SpectralFlux blockSize=%d stepSize=%d' % (blocksize, step))
#fp.addFeature('SpectralRolloff: SpectralRolloff blockSize=%d stepSize=%d' % (blocksize, step))
#fp.addFeature('SpectralSlope: SpectralSlope blockSize=%d stepSize=%d' % (blocksize, step))
#fp.addFeature('SpectralVariation: SpectralVariation blockSize=%d stepSize=%d' % (blocksize, step))
#fp.addFeature('ZCR: ZCR blockSize=%d stepSize=%d' % (blocksize, step))
#fp.addFeature('Energy: Energy blockSize=%d stepSize=%d' % (blocksize, step))
#fp.addFeature('LPC: LPC blockSize=%d stepSize=%d' % (blocksize, step))
#fp.addFeature('MagSpec: MagnitudeSpectrum blockSize=%d stepSize=%d' % (blocksize, step))



# A DataFlow object hold a directed acyclic graph of computational steps describing how to compute some audio features.
df = fp.getDataFlow()
# A Engine object is in charge of processing computations defined in a DataFlow object on given inputs
engine = Engine()
engine.load(df)

# go and process all our files in the dictionaries
afp = AudioFileProcessor()

# SVM formatted data file
import datetime
today = datetime.date.today()
f = open('feature_output_.' +today.ctime()+ '.dat','w')

for dic in diclist:
    print (dic['fname'], dic['class'], dic['path'])
    afp.processFile(engine, dic['path']+dic['fname']) # extract features from an audio file using AudioFileProcessor
    feats = engine.readAllOutputs()


    type = 0 # background =1 foreground =2 backforeground = 3
    if dic['class'] == 'background':
    	type = 1
    elif dic['class'] == 'foreground':
    	type = 2
    elif dic['class'] == 'bafoground':
    	type = 3
	
    features = [] # used to log which feature we are at

    for feat in feats: # go through all the features
        feature = feats[feat]
        accum = None ;
        #print feat + ' ' + str(len(feats[feat]))
        # used for filtering feature to featureNames
        
        
        #if len(feature[0]) > 1:
        accum = zip(*feature)
        count = 1
    	for cc in accum: # go through each dimension
            n = len(cc)
            mean = sum(cc)/n
            std = sqrt(sum((x-mean)**2 for x in cc) / n)
            
            if len(feature[0]) > 1:
            	keyname_mean = feat + str(count) + '_Mean' 
            	keyname_std = feat + str(count) + '_Std'
                count += 1
            else:
            	keyname_mean = feat + '_Mean'
            	keyname_std = feat + '_Std'       
            
            # filter logging by featureNames
            mvalue = mean
            if keyname_mean in featureNames: 
                print keyname_mean
                features.append(mvalue)
            svalue = std
            if keyname_std in featureNames: 
                print keyname_std
                features.append(svalue)
			
    # write current set of features to file
    entry = str(type) + ' '
    for i in range(len(features)):
    	entry += str(i+1)+':'+str(features[i])+' '
    entry = entry[:len(entry)-1] # remove the final space
    f.write(entry+'\n')
f.close() # finito

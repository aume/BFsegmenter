import os
import sys
from math import sqrt
import csv
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

#FeaturePlan is a collection of features to extract, configured for a specific sample rate.
fp = FeaturePlan(sample_rate=44100, resample=False)

fp.addFeature('MFCC: MFCC CepsNbCoeffs=3 blockSize=1024 stepSize=512')
fp.addFeature('Loudness: Loudness LMode=Total blockSize=1024 stepSize=512')
'''
fp.addFeature('PerceptualSharpness: PerceptualSharpness blockSize=512 stepSize=256')
fp.addFeature('PerceptualSpread: PerceptualSpread blockSize=512 stepSize=256')
fp.addFeature('PerceptualFlatnes: PerceptualFlatnes blockSize=512 stepSize=256')

fp.addFeature('SpectralFlux: SpectralFlux blockSize=512 stepSize=256')
fp.addFeature('SpectralRolloff: SpectralRolloff blockSize=512 stepSize=256')
fp.addFeature('SpectralSlope: SpectralSlope blockSize=512 stepSize=256')
fp.addFeature('SpectralVariation: SpectralVariation blockSize=512 stepSize=256')
fp.addFeature('ZCR: ZCR blockSize=512 stepSize=256')
fp.addFeature('Energy: Energy blockSize=512 stepSize=256')
fp.addFeature('LPC: LPC blockSize=512 stepSize=256')
fp.addFeature('MagSpec: MagnitudeSpectrum blockSize=512 stepSize=256')
'''


# A DataFlow object hold a directed acyclic graph of computational steps describing how to compute some audio features.
df = fp.getDataFlow()
# A Engine object is in charge of processing computations defined in a DataFlow object on given inputs
engine = Engine()
engine.load(df)

# go and process all our files in the dictionaries
afp = AudioFileProcessor()
csv_headers = [] # feature line
csv_rows = [] # # a list of processed feature dictionaries


for dic in diclist:
	print (dic['fname'], dic['class'], dic['path'])
	afp.processFile(engine, dic['path']+dic['fname']) # extract features from an audio file using AudioFileProcessor
	feats = engine.readAllOutputs()

	featDic = {} # store the features for a file 
	#featDic['name'] = dic['fname'] 
	featDic['type'] = dic['class']
	
	for feat in feats: # go through all the features
		feature = feats[feat]
		accum = None ;
		
		#if len(feature[0]) > 1:
		accum = zip(*feature)
		count = 1
		for cc in accum: # go through each vector
			n = len(cc)
			mean = sum(cc)/n
			std = sqrt(sum((x-mean)**2 for x in cc) / n)
			# add it to our dictionary
			# if the number of feature dimensions is more than 1 like with mfcc
			if len(feature[0]) > 1:
				keyname_mean = feat + ' mean ' + str(count)
				keyname_std = feat + ' std_dev ' + str(count)
			else:
				keyname_mean = feat + ' mean'
				keyname_std = feat + ' std_dev'
			featDic[keyname_mean] = mean
			featDic[keyname_std] = std
			count += 1
	print featDic
	csv_rows.append(featDic)
	 # create the topline for the csv file if not already
	if len(csv_headers) <= 0:
		for key in sorted(featDic.iterkeys()):
			csv_headers.append(key)
	
	

myWriter = csv.DictWriter(open(sys.argv[1] + '/output.csv', 'wb'), fieldnames=csv_headers)
headers = dict( (n,n) for n in csv_headers )
myWriter.writerow(headers)
# keep floating point accuracy
for dic in csv_rows:
	dic = dict((k, (repr(v) if isinstance(v, float) else str(v))) for k, v in dic.items())
	myWriter.writerow(dic)


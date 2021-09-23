import os
import sys
from math import sqrt


class EssentiaEngine:
    def __init__(self, srate, bsize):
    	#FeaturePlan is a collection of features to extract, configured for a specific sample rate.
    	fp = FeaturePlan(sample_rate=srate, resample=True)
	
    	blocksize = bsize
    	step = bsize/2
        self.numMFCC = 12
    	fp.addFeature('MFCC: MFCC CepsNbCoeffs=%d blockSize=%d stepSize=%d' % (self.numMFCC,blocksize, step))
    	fp.addFeature('Loudness: Loudness LMode=Total blockSize=%d stepSize=%d' % (blocksize, step))
	
    	fp.addFeature('PerceptualSharpness: PerceptualSharpness blockSize=%d stepSize=%d' % (blocksize, step))
    	fp.addFeature('PerceptualSpread: PerceptualSpread blockSize=%d stepSize=%d' % (blocksize, step))
    	#fp.addFeature('PerceptualFlatnes: PerceptualFlatness blockSize=%d stepSize=%d' % (blocksize, step))
	
    	fp.addFeature('SpectralFlux: SpectralFlux blockSize=%d stepSize=%d' % (blocksize, step))
    	fp.addFeature('SpectralRolloff: SpectralRolloff blockSize=%d stepSize=%d' % (blocksize, step))
    	fp.addFeature('SpectralSlope: SpectralSlope blockSize=%d stepSize=%d' % (blocksize, step))
    	fp.addFeature('SpectralVariation: SpectralVariation blockSize=%d stepSize=%d' % (blocksize, step))
    	fp.addFeature('ZCR: ZCR blockSize=%d stepSize=%d' % (blocksize, step))
    	fp.addFeature('Energy: Energy blockSize=%d stepSize=%d' % (blocksize, step))
    	#fp.addFeature('LPC: LPC blockSize=%d stepSize=%d' % (blocksize, step))
    	#fp.addFeature('MagSpec: MagnitudeSpectrum blockSize=%d stepSize=%d' % (blocksize, step))
	
	
	
    	# A DataFlow object hold a directed acyclic graph of computational steps describing how to compute some audio features.
    	df = fp.getDataFlow()
    	# A Engine object is in charge of processing computations defined in a DataFlow object on given inputs
    	self.engine = Engine()
    	self.engine.load(df)
	
    	# go and process all our files in the dictionaries
    	self.afp = AudioFileProcessor()


    # runs the yaafe extraction
    def extractFeatures(self, file):
    	self.afp.processFile(self.engine, file) # extract features from an audio file using AudioFileProcessor
    	feats = self.engine.readAllOutputs()
    	self.engine.flush()
    	return feats

    # # returns an array of feature vectors
    # def featureVectors(self, file, nump=False):
    #     feats = self.extractFeatures(file)
        
    #     flist =[]
    #     fnames = []
    #     for feat in feats:
    #         if feat == 'MFCC':
    #             for i in range(self.numMFCC):
    #                 fnames.append('MFCC'+str(i+1))
    #         else:
    #             fnames.append(feat)
    #         if not nump: flist.append(feats[feat].tolist())
    #         else: flist.append(feats[feat])
    #     flist = zip(*flist)
    #     fvecs = []
    #     for f in flist:
    #         fvecs.append([j for i in f for j in i])
    # 	#print len(fvecs[1])
    # 	return fvecs, fnames
		
	def featureVectors(self, file):
		
import os
import sys
from math import sqrt
import essentia.standard as engine
import essentia.utils as utils
import essentia


class EssentiaEngine:
	def __init__(self, sampleRate, frameSize, hopSize):
		self.sampleRate = sampleRate
		self.frameSize = frameSize
		self.hopSize = hopSize

		# algorithms
		self.extractor = engine.Extractor(lowLevelFrameSize = frameSize, lowLevelHopSize = hopSize, dynamicsFrameSize = frameSize, dynamicsHopSize = hopSize)
		print('engine start')

	# replay gain
	def get_rgain(self, audio):
		return self.rgain(audio)
	
	def get_window(self, frame):
		return self.window(frame)

	def get_spectrum(self, windowedFrame):
		return self.spectrum(windowedFrame)

	def get_spectral_contrast(self, frameSpectrum):
		sc_coeff, sc_valley = self.spectral_contrast(frameSpectrum)
		return sc_valley

	def get_spectral_flux(self, frameSpectrum):
		return self.spectral_flux(frameSpectrum)

	# Gammatone-frequency cepstral coefficients
	def get_gfcc(self, frameSpectrum):
		bands, gfccs = self.gfcc(frameSpectrum)
		return gfccs

	def get_rms(self, frameSpectrum):
		return self.rms(frameSpectrum)

	def get_silence_rate(self, frame, silence_threshold_dB):
		p = essentia.instantPower( frame )
		silence_threshold = pow(10.0, (silence_threshold_dB / 10.0))
		if p < silence_threshold:
			return 1.0
		else:
			return 0.0
	
	def get_mfcc(self, frameSpectrum):
		return self.mfcc(frameSpectrum)

	
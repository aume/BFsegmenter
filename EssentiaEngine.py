import os
import sys
from math import sqrt
import essentia.standard as engine
import essentia.utils as utils


class EssentiaEngine:
	def __init__(self, sampleRate, frameSize):
		self.sampleRate = sampleRate
		self.frameSize = frameSize

		# algorithms
		self.window = engine.Windowing(
		    type='blackmanharris62', zeroPadding=0, size=self.frameSize)
		self.spectral_contrast = engine.SpectralContrast(frameSize=self.frameSize,
                                                        sampleRate=self.sampleRate,
                                                        numberBands=6,
                                                        lowFrequencyBound=20,
                                                        highFrequencyBound=11000,
                                                        neighbourRatio=0.4,
                                                        staticDistribution=0.15)
		self.spectrum = engine.Spectrum(size=frameSize)
		self.silence_rate = engine.SilenceRate(thresholds=[utils.db2lin(-60.0/2.0)])
		self.spectral_flux = engine.Flux()
		self.gfcc = engine.GFCC(sampleRate=self.sampleRate)
		self.rms = engine.RMS()
		self.rgain = engine.ReplayGain(sampleRate=self.sampleRate)

	# replay gain
	def rgain(self, audio):
		return self.rgain(audio)
	
	def window(self, frame):
		return self.window(frame)

	def spectrum(self, windowedFrame):
		return self.engine.spectrum(windowedFrame)

	def spectral_contrast(self, frameSpectrum):
		sc_coeff, sc_valley = self.spectral_contrast(frameSpectrum)
		return sc_valley

	def silence_rate(self, frame):
		return self.silence_rate(frame)

	def spectral_flux(self, frameSpectrum):
		return self.spectral_flux(frameSpectrum)

	# Gammatone-frequency cepstral coefficients
	def gfcc(self, frameSpectrum):
		bands, gfccs = self.gfcc(frameSpectrum)
		return gfccs

	def rms(self, frameSpectrum):
		return self.rms(frameSpectrum)

		
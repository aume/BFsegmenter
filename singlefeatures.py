import essentia.standard as engine
import essentia.utils as utils
import numpy as np
from extractor import getEverything
import essentia

sampleRate = 44100
frameSize = 1024
hopSize = 512

# we start by instantiating the audio loader:
file = 'BF200Corpus/background/aiff/5.aiff'
loader = engine.MonoLoader(filename = file, sampleRate=sampleRate)

# and then we actually perform the loading:
audio = loader()

# create algorithm instances
window = engine.Windowing(type = 'blackmanharris62', zeroPadding = 0, size = frameSize)
spectral_contrast = engine.SpectralContrast(frameSize = frameSize,
                                                  sampleRate = sampleRate,
                                                  numberBands = 6,
                                                  lowFrequencyBound = 20,
                                                  highFrequencyBound = 11000,
                                                  neighbourRatio = 0.4,
                                                  staticDistribution = 0.15)
spectrum = engine.Spectrum(size = frameSize)
# silence rate 
thresholds=[utils.db2lin(val/2.0) for val in [-60.0]]
silence_rate = engine.SilenceRate( thresholds = thresholds )
# spectral flux
spectral_flux = engine.Flux()
# Gammatone-frequency cepstral coefficients
gfcc = engine.GFCC(sampleRate = sampleRate)
# spectral RMS
rms = engine.RMS()
# replay gain
rgain = engine.ReplayGain(sampleRate = sampleRate)

sc_coeffs = []
sc_valleys = []
pool = essentia.Pool()

for frame in engine.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True, lastFrameToEndOfFile = True):
    # spectral contrast valleys
    frame_windowed = window(frame)
    frame_spectrum = spectrum(frame_windowed)
    sc_coeff, sc_valley = spectral_contrast(frame_spectrum)
    pool.add('lowlevel.spectral_contrast_valleys', sc_valley)

    # silence rate
    pool.add('lowlevel.silence_rate', silence_rate(frame))

    # spectral flux
    pool.add('lowlevel.spectral_flux', spectral_flux(frame_spectrum))

    # Gammatone-frequency cepstral coefficients
    bands, gfccs = gfcc(frame_spectrum)
    pool.add('lowlevel.gfcc', gfccs)

    # spectral RMS
    pool.add('lowlevel.spectral_rms', rms(frame_spectrum))


# replay gain
pool.add('replay_gain', rgain(audio))

# compute mean and variance of the frames
aggrPool = engine.PoolAggregator(defaultStats = [ 'mean', 'stdev' ])(pool)

print(aggrPool.descriptorNames())
print(aggrPool['lowlevel.spectral_contrast_valleys.mean'])
print(aggrPool['lowlevel.silence_rate.stdev'])
print(aggrPool['lowlevel.spectral_flux.mean'])
print(aggrPool['lowlevel.gfcc.mean'])
print(aggrPool['lowlevel.spectral_rms.mean'])
print(pool['replay_gain'])

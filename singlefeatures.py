import essentia.standard as engine
import essentia.utils as utils
import essentia 

sampleRate = 44100
frameSize = 1024
hopSize = 512
windowDuration = 0.5 # analysis window length in seconds

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
# create pool for storage and aggregation
pool = essentia.Pool()
# frame counter used to detect end of window
frameCount = 0
# calculate the length of analysis frames
frame_duration = float(frameSize / 2)/float(sampleRate)
num_frames = int(windowDuration / frame_duration) # number frames in a window

for frame in engine.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True, lastFrameToEndOfFile = True):
    # replay gain TODO
    pool.add('replay_gain', rgain(audio))

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

    frameCount += 1

    # detect if we have traversed a whole window
    if (frameCount == num_frames):
        print('\nWINDOW')
        # compute mean and variance of the frames
        aggrPool = engine.PoolAggregator(defaultStats = [ 'mean', 'stdev' ])(pool)
        correlationAttributes = {}
        correlationAttributes['lowlevel.silence_rate.stdev'] = aggrPool['lowlevel.silence_rate.stdev']
        correlationAttributes['lowlevel.spectral_contrast_valleys.mean.0'] = aggrPool['lowlevel.spectral_contrast_valleys.mean'][0]
        correlationAttributes['replay_gain'] = pool['replay_gain'][0]
        correlationAttributes['lowlevel.spectral_contrast_valleys.stdev.2'] = aggrPool['lowlevel.spectral_contrast_valleys.stdev'][2]
        correlationAttributes['lowlevel.spectral_contrast_valleys.stdev.3'] = aggrPool['lowlevel.spectral_contrast_valleys.stdev'][3]
        correlationAttributes['lowlevel.spectral_contrast_valleys.stdev.4'] = aggrPool['lowlevel.spectral_contrast_valleys.stdev'][4]
        correlationAttributes['lowlevel.spectral_contrast_valleys.stdev.5'] = aggrPool['lowlevel.spectral_contrast_valleys.stdev'][5]
        correlationAttributes['lowlevel.spectral_flux.mean'] = aggrPool['lowlevel.spectral_rms.mean']
        correlationAttributes['lowlevel.gfcc.mean.0'] = aggrPool['lowlevel.gfcc.mean'][0]
        correlationAttributes['lowlevel.spectral_rms.mean'] = aggrPool['lowlevel.spectral_rms.mean']

        print(correlationAttributes)
        # reset counter and clear pool
        frameCount = 0
        pool.clear()
        aggrPool.clear()





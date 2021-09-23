import essentia.standard as engine
import essentia.utils as utils


# we start by instantiating the audio loader:
file = directory + "/" + fileName
loader = engine.MonoLoader(filename = file)

# and then we actually perform the loading:
audio = loader()

# get 60 db silence rate
thresh = utils.db2lin(-60/2.0)
silenceVector = []

for frame in engine.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
    sr = engine.SilenceRate(thresholds=thresh)
    silenceVector.append(sr)
    # mfcc_bands, mfcc_coeffs = mfcc(engine.spectrum(w(frame)))
    # mfccs.append(mfcc_coeffs)
    # melbands.append(mfcc_bands)
    # melbands_log.append(logNorm(mfcc_bands))

def getSilenceRate(frame):
    sr = engine.SilenceRate(thresholds=thresh)
    silenceRate = sr(frame)
    return silenceRate
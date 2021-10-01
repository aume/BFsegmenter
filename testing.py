from segmenter import Segmenter

s1 = Segmenter()

file = 'BF200Corpus/background/aiff/5.aiff'

v = s1.regionsChunk(file)
print(v)
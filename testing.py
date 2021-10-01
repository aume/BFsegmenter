from segmenter import Segmenter

s1 = Segmenter()

file = 'BF90Corpus/Validation/background/aiff/1.aiff'

v = s1.regionsChunk(file)
# print(v)
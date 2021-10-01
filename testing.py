from segmenter import Segmenter

s1 = Segmenter()

file = 'BF90Corpus/Validation/foreground/aiff/9.aiff'

v = s1.regionsChunk(file)
# print(v)
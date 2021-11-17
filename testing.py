from segmenter import Segmenter

s1 = Segmenter()

file = 'BF200Corpus/bafoground/aiff/34.aiff'

windowData = s1.segment(file)

for item in windowData:
    print('\n',item)

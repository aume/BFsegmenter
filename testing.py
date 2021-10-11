from segmenter import Segmenter

s1 = Segmenter()

file = 'BF90Corpus/Validation/foreground/aiff/9.aiff'

windowData = s1.segment(file)

for item in windowData:
    print(item)

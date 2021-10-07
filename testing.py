from segmenter import Segmenter

s1 = Segmenter()

file = 'BF90Corpus/Validation/foreground/aiff/9.aiff'

windowData = s1.regionsChunk(file)

print('\nbefore')
for item in windowData:
    print(item['type'])

filteredWindowData = s1.Clustering(windowData)

print('\nafter')
for item in filteredWindowData:
    print(item['type'])
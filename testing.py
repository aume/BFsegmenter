from segmenter import Segmenter

s1 = Segmenter()

file = 'BF90Corpus/Validation/foreground/aiff/9.aiff'

windowData = s1.regionsChunk(file)

print('\nbefore')
for item in windowData:
    print('\n',item)

filteredWindowData = s1.Clustering(windowData)

print('\nCluster')
for item in filteredWindowData:
    print('\n',item)

conjunctData = s1.conjunction(filteredWindowData)

print('\nConjuction')
for item in conjunctData:
    print(item)
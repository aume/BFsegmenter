from segmenter import Segmenter
import os

s1 = Segmenter()

for filename in os.listdir('Test6Sounds'):
    print('\nrunning: ', filename)
    directory = 'Test6Sounds/' + filename
    windowData = s1.segment(directory)
    for item in windowData:
        print('\n',item)

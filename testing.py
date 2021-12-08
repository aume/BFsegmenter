from segmenter import Segmenter
import os
import numpy as np

s1 = Segmenter()

foldername = 'TestSounds'

for filename in os.listdir(foldername):
    print('\nrunning: ', filename)
    path = foldername + '/' + filename
    windowData = s1.segment(path)
    for item in windowData:
        # print('\n %f %f %f %s\t arousal: %f valence: %f'% (item['start'], item['end'], item['duration'], item['type'], item['arousal'], item['valence']))
        print('%f %f %f %s'% (item['start'], item['end'], item['duration'], item['type']))
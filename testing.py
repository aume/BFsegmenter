from segmenter import Segmenter
import os

s1 = Segmenter()

foldername = 'TestSounds'

for filename in os.listdir(foldername):
    print('\nrunning: ', filename)
    directory = foldername + '/' + filename
    windowData = s1.segment(directory)
    for item in windowData:
        print('\n %f %f %f %s'% (item['start'], item['end'], item['duration'], item['type']))

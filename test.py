from pydub import AudioSegment
import os

file = '52_bg.aiff'
filename = os.path.basename(file)
corpusName = 'SegmentedCorpus'  # dir to put segments

print(filename)
song = AudioSegment.from_file(filename, 'aiff')
output_folder = './' + corpusName + '/'

recregion = song[0.0 * 1000:1.0 * 1000]  # cut the region
awesome = recregion.fade_in(50).fade_out(50)  # fade in/out
awesome.export(output_folder + 'r_11.aiff', format='aiff')  # save to disk

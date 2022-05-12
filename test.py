from pydub import AudioSegment

file = 'PotteryInterview.wav'
corpusName = 'SegmentedCorpus'  # dir to put segments

song = AudioSegment.from_wav(file)
output_folder = './' + corpusName + '/'

recregion = song[0.0 * 1000:1.0 * 1000]  # cut the region
awesome = recregion.fade_in(50).fade_out(50)  # fade in/out
awesome.export(output_folder + 'r_11', format='wav')  # save to disk

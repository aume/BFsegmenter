#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
corpusAquire. Script to segment and cut an audio file.
Cut file is saved to disk and results logged in sql db.
Can generate a corpus a local directory
Use
python corpusAquireDir.py /path/to/audio/files
audio files are aif or wav
segments the files in the directory and puts the segs into SegmentedCorpus
'''

from segmenter import Segmenter
import sys
from random import choice
import os
from pydub import AudioSegment
import sqlite3 as lite

# connect to the database
con = lite.connect('./corpus.db')

# audio file regions filtered and trimmed to this duration
desiredDuration = 4.0
corpusName = 'SegmentedCorpus'  # dir to put segments
# searchDirectory = sys.argv[1]  # search this directory
searchDirectory = 'soundfiles'  # search this directory

# init the segmenter and train with BF Corpus
segmenter = Segmenter()
output_folder = './' + corpusName + '/'

# create output folder if it doesnt exist
if not os.path.exists(corpusName):
    os.makedirs(corpusName)

# method that segments file
def process_file(file, id):
    
    print('processing file')

    rli = segmenter.segment(file)  # segment the file

    candidates = []  # all regions of desired length

    # go through the regions for this file
    for i in rli[1]:
        if i[1] >= desiredDuration:
            if i[1] > desiredDuration + 0.1:  # trim out the middle if its longer
                diff = i[1] - desiredDuration
                i[2] = i[2] + diff / 2
                i[3] = i[3] - diff / 2
                i[1] = i[3] - i[2]
            candidates.append(i)

    if len(candidates):  # if there is more than none
        region = choice(candidates)  # choose one regions randomly
        filename = os.path.basename(file)

        good = True

        try:
            song = AudioSegment.from_file(file, 'aiff')  # open the file
        except EOFError:

            # good = False

            print('Maybe wav')
            try:
                song = AudioSegment.from_file(file, 'wav')  # open the file
            except EOFError:
                good = False
                print('BAD')

        if good:
            recregion = song[region[2] * 1000:region[3] * 1000]  # cut the region
            awesome = recregion.fade_in(50).fade_out(50)  # fade in/out
            awesome.export(output_folder + 'r_' + filename,
                           format='aiff')  # save to disk

            filename = os.path.basename(file)
            (feats, type) = segmenter.features_only(output_folder + 'r_'
                     + filename)

            print('r_' + filename)
            print(region[1])
            print(region[0])
            # print('Loud Mean = ' + str(feats[0]))
            # print('Loud Std = ' + str(feats[1]))
            # print('MFCC1 Mean = ' + str(feats[2]))
            # print('MFCC1 Std = ' + str(feats[3]))
            # print('MFCC2 Mean = ' + str(feats[4]))
            # print('MFCC2 Std = ' + str(feats[5]))
            # print('MFCC3 Mean = ' + str(feats[6]))
            # print('MFCC3 Std = ' + str(feats[7]))

            # # save the data entry id, name, duration, class, features
            # with con:
            #     try:
            #         cur.execute('INSERT INTO ' + corpusName
            #                     + ' (FsID, FileName, Duration, Class, Loud_Mean, Loud_Std,MFCC1_Mean, MFCC1_Std, MFCC2_Mean, MFCC2_Std, MFCC3_Mean, MFCC3_Std) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
            #                     , (
            #             id,
            #             'r_' + filename,
            #             desiredDuration,
            #             region[0],
            #             feats[0],
            #             feats[1],
            #             feats[2],
            #             feats[3],
            #             feats[4],
            #             feats[5],
            #             feats[6],
            #             feats[7],
            #             ))
            #     except:
            #         print('not inserted')


# make a new table to store all the data
# with con:
#     cur = con.cursor()
#     try:
#         cur.execute('CREATE TABLE ' + corpusName
#                     + ' (Id INTEGER PRIMARY KEY, FsID INTEGER, FileName TEXT, Duration FLOAT, Class TEXT, Loud_Mean FLOAT, Loud_Std FLOAT, MFCC1_Mean FLOAT, MFCC1_Std FLOAT, MFCC2_Mean FLOAT, MFCC2_Std FLOAT, MFCC3_Mean FLOAT, MFCC3_Std FLOAT)'
#                     )
#     except:
#         print('already got the table')

# process_file("/Users/timmy/Desktop/temp/batchtest/3288_2518-hq.aiff")




fid = 0
for (root, dirs, files) in os.walk(searchDirectory):
    path = root.split('/')

    # print((len(path) - 1) *'---' , os.path.basename(root)

    for file in files:
        if file.endswith('.aif') or file.endswith('.aiff') \
            or file.endswith('.wav'):
            print(len(path) * '---', root, file)
            process_file(root + '/' + file, fid)
            fid += 1






# with con:
#     cur.execute('SELECT * FROM ' + corpusName + '')

#     rows = cur.fetchall()

#     for row in rows:
#         print(row)
import matplotlib.patches as patches
import essentia.standard
from segmenter import Segmenter
import os
import numpy as np

from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt
# set plot sizes to something larger than default
plt.rcParams['figure.figsize'] = (15, 6)

# program for testing and tuning segmentation
sample_rate = 22500


def main():
    s1 = Segmenter()
    foldername = 'TestSounds'

    for filename in os.listdir(foldername):
        print('\nrunning: ', filename)
        path = foldername + '/' + filename
        windowData = s1.segment(path)
        segs = []
        for item in windowData:
            print(item)
            # print('\n %f %f %f %s\t arousal: %f valence: %f'% (item['start'], item['end'], item['duration'], item['type'], item['arousal'], item['valence']))
            print('%f %f %f %s %s' % (
                item['start'], item['end'], item['duration'], item['type'], item['probabilities']))
            segs.append([item['start'], item['duration'], item['type']])
        plotaudio(filename, path, segs)


def plotaudio(filename, path, segments):
    foreground_color = '#ff9191'  # red
    background_color = '#91deff'  # blue
    bafoground_color = '#fff7cc'  # yellow

    print('plotting audio')
    loader = essentia.standard.MonoLoader(
        filename=path, sampleRate=sample_rate)
    audio = loader()

    fig, ax = plt.subplots()

    shortCategoryNames = {
        'fore': 'f',
        'back': 'b',
        'backfore': 'bf'
    }

    for segment in segments:
        # pull data from segment
        start = segment[0] * sample_rate
        dur = segment[1] * sample_rate
        category = segment[2]
        # determine color
        if category == 'fore':
            color = foreground_color
        elif category == 'back':
            color = background_color
        else:
            color = bafoground_color
        # Create a Rectangle patch
        rect = patches.Rectangle((start, -1), dur, 2, linewidth=1, color=color)
        ax.add_patch(rect)
        ax.text(start, 0.95, shortCategoryNames[category])

    plot(audio[:])
    plt.title('Displaying audio: %s' % filename)
    show()


if __name__ == '__main__':
    main()

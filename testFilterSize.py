from segmenter import Segmenter
import os
import numpy as np

from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (30, 15) # set plot sizes to something larger than default
import essentia.standard
import matplotlib.patches as patches

# program for testing and tuning segmentation

sample_rate = 22500
maxFilterWindow = 8 #choose 2 or higher


def main():
    s1 = Segmenter()
    foldername = 'TestSounds'

    filterRange = range(0, maxFilterWindow)

    for filename in os.listdir(foldername):
        segversions = []
        for filterwindow in filterRange:
            print('\nrunning: ', filename)
            path = foldername + '/' + filename
            s1.filterWindow = filterwindow
            windowData = s1.segment(path)
            segs = []
            for item in windowData:
                # print('\n %f %f %f %s\t arousal: %f valence: %f'% (item['start'], item['end'], item['duration'], item['type'], item['arousal'], item['valence']))
                print('%f %f %f %s %s'% (item['start'], item['end'], item['duration'], item['type'], item['probabilities']))
                segs.append([item['start'], item['duration'], item['type']])
            segversions.append(segs)
        plotaudio(filename, path, segversions)

def plotaudio(filename, path, segversions):
    foreground_color = '#ff9191' #red
    background_color = '#91deff' #blue
    bafoground_color = '#fff7cc' #yellow

    

    print('plotting audio')
    loader = essentia.standard.MonoLoader(filename=path, sampleRate = sample_rate)
    audio = loader()
    fig = plt.figure()
    gs = fig.add_gridspec(maxFilterWindow, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)

    shortCategoryNames = {
        'fore':'f',
        'back':'b',
        'backfore':'bf'
    }
    for index, version in enumerate(segversions):
        for segment in version:
            # pull data from segment
            start = segment[0] * sample_rate
            dur = segment[1] * sample_rate
            category = segment[2]
            print('start %s' % start)
            print('dur %s' % dur)
            print('category %s' % category)
            # determine color
            if category == 'fore':
                color = foreground_color
            elif category == 'back':
                color = background_color
            else:
                color = bafoground_color
            # Create a Rectangle patch
            rect = patches.Rectangle((start, -1), dur, 2, linewidth=1, color=color)
            axs[index].add_patch(rect)
            axs[index].text(start, 0.85, shortCategoryNames[category])
            axs[index].plot(audio[:], color='b')
            axs[index].set_title('Filter window: ' + str(index), loc='left')

    # plt.title('Displaying audio: %s' % filename)
    show()

if __name__ == '__main__':
    main()


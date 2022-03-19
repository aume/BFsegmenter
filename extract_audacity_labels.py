from segmenter import Segmenter
import os

# program for testing and tuning segmentation
sample_rate = 22500
foldername = 'soundstorage'

def main():
    s1 = Segmenter()
    for filename in os.listdir(foldername):
        path = foldername + '/' + filename
        windowData = s1.segment(path)
        
        label_string = ''
        for item in windowData:
            label_string += str(item['start'])+'\t'+str(item['end'])+'\t'+str(item['type'])+'\n'
    
        # open text file and write label data
        if('.mp3' in filename):
            labelname = filename.replace('.mp3', '_MPclusering.txt')
        elif('.wav' in  filename):
            labelname = filename.replace('.wav', '_MPclusering.txt')
        f = open('label_output/' + labelname, 'w')
        f.write(label_string)
        f.close()

if __name__ == '__main__':
    main()

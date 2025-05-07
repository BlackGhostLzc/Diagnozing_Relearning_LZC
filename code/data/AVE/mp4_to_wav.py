import os


train_videos = '/root/autodl-tmp/AVE_Dataset/trainSet.txt'
test_videos = '/root/autodl-tmp/AVE_Dataset/testSet.txt'


wave_files_dir = '/root/autodl-tmp/AVE_Dataset/wave_files'
os.mkdir(wave_files_dir)


# test set processing
with open(test_videos, 'r') as f:
    files = f.readlines()

for i, item in enumerate(files):
    if i % 500 == 0:
        print('*******************************************')
        print('{}/{}'.format(i, len(files)))
        print('*******************************************')
    item = item.split('&')
    mp4_filename = os.path.join('/root/autodl-tmp/AVE_Dataset/AVE', item[1]+'.mp4')
    wav_filename = os.path.join(wave_files_dir, item[1]+'.wav')

    if os.path.exists(wav_filename):
        pass
    else:
        os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))

# train set processing
with open(train_videos, 'r') as f:
    files = f.readlines()

for i, item in enumerate(files):
    if i % 500 == 0:
        print('*******************************************')
        print('{}/{}'.format(i, len(files)))
        print('*******************************************')
    item = item.split('&')
    mp4_filename = os.path.join('/root/autodl-tmp/AVE_Dataset/AVE', item[1]+'.mp4')
    wav_filename = os.path.join(wave_files_dir, item[1]+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))

import csv
import os

import numpy as np
import torchaudio
import torch

## save path of processed spectrogram
# train_npy_dir = '/root/autodl-tmp/AVE_Dataset/train_audio_npy'
# test_npy_dir = '/root/autodl-tmp/AVE_Dataset/test_audio_npy'
# os.mkdir(train_npy_dir)
# os.mkdir(test_npy_dir)
npy_dir = '/root/autodl-tmp/AVE_Dataset/audio_npy_files'
os.mkdir(npy_dir)

## file path of wav files
wav_dir = '/root/autodl-tmp/AVE_Dataset/wave_files'


## the list of all wav files
train_csv_file = '/root/autodl-tmp/AVE_Dataset/trainSet.txt'
test_csv_file = '/root/autodl-tmp/AVE_Dataset/testSet.txt'

data = []
with open(train_csv_file) as f:
    for line in f:
        item = line.split('&')
        name = item[1]
        wav_file = os.path.join(wav_dir, name + '.wav')
        if os.path.exists(wav_file):
            data.append(name)

with open(test_csv_file) as f:
    for line in f:
        item = line.split('&')
        name = item[1]
        wav_file = os.path.join(wav_dir, name + '.wav')
        if os.path.exists(wav_file):
            data.append(name)


def process(data, wav_dir, npy_dir):
    for name in data:
        waveform, sr = torchaudio.load(wav_dir + '/'+ name + '.wav')
        waveform = waveform - waveform.mean()
        norm_mean = -4.503877
        norm_std = 5.141276

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

        target_length = 1024
        n_frames = fbank.shape[0]
        # print(n_frames)
        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        fbank = (fbank - norm_mean) / (norm_std * 2)

        print(fbank.shape)
        np.save(npy_dir + '/' + name + '.npy', fbank)
        print(npy_dir + '/' + name + '.npy')

process(data, wav_dir, npy_dir)

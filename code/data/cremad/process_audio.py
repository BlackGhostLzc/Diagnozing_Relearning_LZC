import csv
import os

import numpy as np
import torchaudio
import torch
import pandas as pd

path_to_dataset = '/root/autodl-tmp/CREMA-D'
csv_file = pd.read_csv(os.path.join(path_to_dataset, 'processedResults/summaryTable.csv'))
audio_path = os.path.join(path_to_dataset, 'AudioWAV')
save_path = os.path.join(path_to_dataset, 'audio_npy_files')
if not os.path.exists(save_path):
    os.mkdir(save_path)

file_list = list(csv_file['FileName'])
print(len(file_list))
file_not_exists = 0
file_exists = 0

for name in file_list:
    wavefile = audio_path + '/' + name + '.wav'
    if not os.path.exists(wavefile):
        # print("不存在该文件")
        file_not_exists+=1
        continue
    waveform, sr = torchaudio.load(wavefile)
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

    print(fbank.shape)                               # torch.Size([1024, 128])
    np.save(save_path + '/' + name + '.npy', fbank)
    file_exists += 1
#
# print(f"有{file_not_exists}个文件不存在")
# print(f"用{file_exists}个文件")

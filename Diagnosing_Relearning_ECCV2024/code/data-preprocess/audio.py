import csv
import os

import numpy as np
import torchaudio
import torch

# 修改点 1: 保存路径后缀明确为 .pt
save_path = './dataset/audio/'  # 建议新建目录区分 .pt 文件

# 其他路径不变
audio_path = '../../CREMA-D/AudioWAV'
csv_file = './dataset/cremad/my_train.csv'

data = []
with open(csv_file) as f:
    for line in f:
        item = line.split("\n")[0].split(" ")
        name = item[0][:-4]
        if os.path.exists(audio_path + '/' + name + '.wav'):
            print(name)
            data.append(name)

for name in data:
    waveform, sr = torchaudio.load(audio_path + '/' + name + '.wav')
    waveform = waveform - waveform.mean()
    norm_mean = -4.503877
    norm_std = 5.141276

    # 生成 fbank 特征（PyTorch Tensor）
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr,
        use_energy=False, window_type='hanning',
        num_mel_bins=128, dither=0.0, frame_shift=10
    )

    # 填充/截断逻辑不变
    target_length = 1024
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    fbank = (fbank - norm_mean) / (norm_std * 2)

    # 修改点 2: 使用 torch.save 保存为 .pt 文件
    print("save path")
    print(os.path.join(save_path, f"{name}.pt"))
    torch.save(fbank, os.path.join(save_path, f"{name}.pt"))  # 直接保存 PyTorch Tensor
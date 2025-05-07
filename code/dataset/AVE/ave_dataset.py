import copy
import csv
import os
# import pickle

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np
class AVEDataset(Dataset):

    def __init__(self, mode='train'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode

        self.data_root = '/root/autodl-tmp/AVE_Dataset'
        self.audio_feature_path = '/root/autodl-tmp/AVE_Dataset/audio_npy_files'

        self.visual_feature_path = os.path.join(self.data_root, 'Image-01-FPS')

        self.stat_path = os.path.join(self.data_root, 'Annotations.txt')
        self.train_txt = os.path.join(self.data_root, 'trainSet.txt')
        self.test_txt = os.path.join(self.data_root, 'testSet.txt')

        with open(self.stat_path) as f1:
            for line in f1:
                line = line.split('&')
                c = line[0]
                if c == 'Category':
                    continue
                if c not in classes:
                    classes.append(c)

        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(csv_file) as f2:
            for line in f2:
                line = line.split('&')
                #print(os.path.join(self.audio_feature_path, item[1] + '.npy'))
                audio_path = os.path.join(self.audio_feature_path, line[1] + '.npy')
                visual_path = os.path.join(self.visual_feature_path, line[1])
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    # AVE, delete repeated labels
                    a = set(data)
                    if line[1] in a:
                        del data2class[line[1]]
                        data.remove(line[1])
                    data.append(line[1])
                    data2class[line[1]] = line[0]
                else:
                    continue

        self.classes = sorted(classes)

        print(self.classes)
        self.data2class = data2class

        self.av_files = []
        for item in data:
            self.av_files.append(item)
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))

    def __len__(self):
        return len(self.av_files)

    def __getitem__(self, idx):
        av_file = self.av_files[idx]
        # print(av_file)
        # print(os.path.join(self.audio_feature_path, av_file + '.npy'))
        # Audio
        audio_path = os.path.join(self.audio_feature_path, av_file + '.npy')
        #spectrogram = pickle.load(open(audio_path, 'rb'))
        spectrogram = np.load(audio_path)
        spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)
        # Visual
        visual_path = os.path.join(self.visual_feature_path, av_file)
        file_num = len(os.listdir(visual_path))

        if self.mode == 'train':

            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        pick_num = 3
        # seg = int(file_num / pick_num)
        # path1 = []
        image = []
        image_arr = []
        # t = [0] * pick_num

        # for i in range(pick_num):
        #     t[i] = seg * i + 1
        #     path1.append('frame_0000' + str(t[i]) + '.jpg')
        #     image.append(Image.open(visual_path + "/" + path1[i]).convert('RGB'))
        #     image_arr.append(transform(image[i]))
        #     image_arr[i] = image_arr[i].unsqueeze(1).float()
        #     if i == 0:
        #         image_n = copy.copy(image_arr[i])
        #     else:
        #         image_n = torch.cat((image_n, image_arr[i]), 1)

        jpg_files = [f for f in os.listdir(visual_path) if f.lower().endswith('.jpg')]
        selected = random.sample(jpg_files, pick_num)
        for i, file in enumerate(selected):
            filepath = os.path.join(visual_path, file)
            image.append(Image.open(filepath).convert('RGB'))
            image_arr.append(transform(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i == 0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)
        # print("spectrogram")
        # print(type(spectrogram))
        # print("image_n")
        # print(type(image_n))
        # print("index")
        # print(self.classes.index(self.data2class[av_file]))
        # return spectrogram, image_n, self.classes.index(self.data2class[av_file]), av_file
        return spectrogram, image_n, self.classes.index(self.data2class[av_file])
import csv
import math
import os
import random
import copy
import numpy as np
import torch
import torch.nn.functional
import torchaudio
from PIL import Image
from scipy import signal
from torch.utils.data import Dataset
from torchvision import transforms




class AVDataset_CD(Dataset):
  def __init__(self, mode='train'):
    classes = []
    self.data = []
    data2class = {}

    self.mode=mode
    ## replace with your visual_path and audio_path
    self.visual_path = '/root/autodl-tmp/CREMA-D/Image-01/'
    self.audio_path = '/root/autodl-tmp/CREMA-D/audio_npy_files/'

    
    self.stat_path = './dataset/cremad/stat.csv'
    self.train_txt = './dataset/cremad/train.csv'
    self.my_train_txt = './dataset/cremad/my_train.csv'
    self.test_txt = './dataset/cremad/test.csv'
    self.val_txt='./dataset/cremad/val.csv'
    if mode == 'train':
        csv_file = self.train_txt
    elif mode=='my_train':
        csv_file = self.my_train_txt
    elif mode=='val':
        csv_file = self.val_txt
    else:
        csv_file = self.test_txt

    
    with open(self.stat_path, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])
    
    with open(csv_file) as f:
      csv_reader = csv.reader(f)
      for item in csv_reader:
        # print(self.audio_path + item[0] + '.npy')
        if item[1] in classes and os.path.exists(self.audio_path + item[0] + '.npy') and os.path.exists(
                        self.visual_path + '/' + item[0]):
            self.data.append(item[0])
            data2class[item[0]] = item[1]

    print('data load over')
    print(len(self.data))
    
    self.classes = sorted(classes)

    self.data2class = data2class
    self._init_atransform()
    print('# of files = %d ' % len(self.data))
    print('# of classes = %d' % len(self.classes))

    #Audio
    self.class_num = len(self.classes)

  def _init_atransform(self):
    self.aid_transform = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
    return len(self.data)

  
  def __getitem__(self, idx):
    datum = self.data[idx]

    # Audio
    fbank = np.load(self.audio_path + datum + '.npy')
    fbank = torch.from_numpy(fbank).unsqueeze(0)
    #fbank = torch.load(self.audio_path + datum + '.npy').unsqueeze(0)

    # Visual
    if self.mode == 'train':
        transf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transf = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # folder_path = self.visual_path + datum
    # # file_num = len(os.listdir(folder_path))
    # pick_num = 1
    # # seg = int(file_num/pick_num)
    # image_arr = []

    # jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    # selected = random.sample(jpg_files, pick_num)
    # for i, file in enumerate(selected):
    #     filepath = os.path.join(folder_path, file)
    #     image_arr.append(transf(Image.open(filepath).convert('RGB')).unsqueeze(0))
    #
    # # for i in range(pick_num):
    # #   if self.mode == 'train':
    # #     index = i * 29
    # #   else:
    # #     index = 0
    # #   path = folder_path + '/000' + str(index).zfill(2) + '.jpg'
    # #   # print(path)
    # #   image_arr.append(transf(Image.open(path).convert('RGB')).unsqueeze(0))

    # [2, 3, 224, 224]
    # images = torch.cat(image_arr)

    folder_path = self.visual_path + datum
    file_num = len(os.listdir(folder_path))
    pick_num = 2
    seg = int(file_num / pick_num)
    image_arr = []

    for i in range(pick_num):
        if self.mode == 'train':
            index = random.randint(i * seg + 1, i * seg + seg)
        else:
            index = i * seg + int(seg / 2)
        path = folder_path + '/000' + str(index).zfill(2) + '.jpg'
        # print(path)
        image_arr.append(transf(Image.open(path).convert('RGB')).unsqueeze(0))

    images = torch.cat(image_arr)

    return fbank, images, self.classes.index(self.data2class[datum])

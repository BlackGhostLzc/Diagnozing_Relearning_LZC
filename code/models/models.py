import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18


class RGBClassifier(nn.Module):
    def __init__(self, args):
        super(RGBClassifier, self).__init__()

        n_classes = 101

        self.visual_net = resnet18(modality='visual')
        self.visual_net.load_state_dict(torch.load('/root/project/resnet18.pth'), strict=False)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, visual):
        B = visual.size()[0]
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        v = F.adaptive_avg_pool3d(v, 1)

        v = torch.flatten(v, 1)

        out = self.fc(v)

        return out





class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))


        self.dataset = args.dataset

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        self.head = nn.Linear(1024, n_classes)
        self.audio_fc = nn.Linear(512, 1024)
        self.video_fc = nn.Linear(512, 1024)
        self.d_a = 512
        self.d_v = 512



    def forward(self, audio, visual):
        if self.dataset != 'CREMAD':
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()

        # visual: [batchSize, pickNum=2, channel=3, w=224, h=224]
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)                  #[64, 512]

        # Added
        '''
        这里的a,v都是512的维度,我们要对这个维度进行更改
        '''
        a = self.audio_fc(a)[:, :self.d_a]
        v = self.video_fc(v)[:, :self.d_v]                   #[64, 1024]

        out = torch.cat((a,v),1)                 #[64, 1024]
        out = self.head(out)

        return out,_,_,a,v


    def update_dimension_av(self, purity_a, purity_b):
        self.d_a = int(purity_a/(purity_b + purity_a) * 1024)
        self.d_v = 1024 - self.d_a

        print(f"更新audio的输出维度为: {self.d_a}")
        print(f"更新visual的输出维度为: {self.d_v}")

        assert self.d_a + self.d_v == 1024
        assert self.d_a > 0
        assert self.d_v > 0
    





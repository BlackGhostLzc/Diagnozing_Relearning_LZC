import pickle

import librosa
import pandas as pd
import cv2
import os
import pdb
import numpy as np
from scipy import signal


class videoReader(object):
    def __init__(self, video_path, frame_interval=1, frame_kept_per_second=1):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.frame_kept_per_second = frame_kept_per_second

        # pdb.set_trace()
        self.vid = cv2.VideoCapture(self.video_path)
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.video_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_len = int(self.video_frames / self.fps)

    def video2frame(self, frame_save_path):
        self.frame_save_path = frame_save_path
        success, image = self.vid.read()
        count = 0
        while success:
            count += 1
            if count % self.frame_interval == 0:
                save_name = '{}/frame_{}_{}.jpg'.format(self.frame_save_path, int(count / self.fps),
                                                        count)  # filename_second_index
                cv2.imencode('.jpg', image)[1].tofile(save_name)
            success, image = self.vid.read()

    def video2frame_update(self, frame_save_path, min_save_frame=3):
        self.frame_save_path = frame_save_path

        count = 0
        save_count = 0
        frame_interval = int(self.fps / self.frame_kept_per_second)
        while count < self.video_frames:
            ret, image = self.vid.read()
            if not ret:
                break
            if count % self.fps == 0:
                frame_id = 0
            if frame_id < frame_interval * self.frame_kept_per_second and frame_id % frame_interval == 0:
                save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
                cv2.imencode('.jpg', image)[1].tofile(save_name)
                save_count += 1

            frame_id += 1
            count += 1

        if save_count < min_save_frame:
            add_count = min_save_frame - save_count
            count = 0
            if self.video_frames < min_save_frame:
                while count < add_count:
                    frame_id = np.random.randint(0, min_save_frame)
                    save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, frame_id)
                    if not os.path.exists(save_name):
                        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, image = self.vid.read()
                        cv2.imencode('.jpg', image)[1].tofile(save_name)
                        count += 1
            else:
                while count < add_count:
                    frame_id = np.random.randint(0, self.video_frames)
                    save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, frame_id)
                    if not os.path.exists(save_name):
                        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                        ret, image = self.vid.read()
                        cv2.imencode('.jpg', image)[1].tofile(save_name)
                        count += 1

    def video2frame_update_SE(self, frame_save_path, min_save_frame=3, start_t=0, end_t=10):
        self.frame_save_path = frame_save_path

        count = 0
        save_count = 0
        frame_interval = int(self.fps / self.frame_kept_per_second)

        num_count = 0
        while count < self.video_frames:
            ret, image = self.vid.read()
            if not ret:
                break
            if count % self.fps == 0:
                frame_id = 0
            if frame_id < frame_interval * self.frame_kept_per_second and frame_id % frame_interval == 0:
                if start_t <= num_count <= end_t:
                    # print('save')
                    save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
                    cv2.imencode('.jpg', image)[1].tofile(save_name)
                    save_count += 1
                num_count += 1

            frame_id += 1
            count += 1

        if save_count < min_save_frame:
            add_count = min_save_frame - save_count
            count = 0
            if self.video_frames < min_save_frame:
                while count < add_count:
                    frame_id = np.random.randint(0, min_save_frame)
                    save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, frame_id)
                    if not os.path.exists(save_name):
                        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, image = self.vid.read()
                        cv2.imencode('.jpg', image)[1].tofile(save_name)
                        count += 1
            else:
                while count < add_count:
                    frame_id = np.random.randint(start_t*self.fps, end_t*self.fps)
                    save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, frame_id)
                    if not os.path.exists(save_name):
                        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                        ret, image = self.vid.read()
                        cv2.imencode('.jpg', image)[1].tofile(save_name)
                        count += 1

    def video2frame_per_second(self, frame_save_path):
        self.frame_save_path = frame_save_path
        if not os.path.exists(frame_save_path):
            os.makedirs(frame_save_path)

        count = 0  # 视频中的帧计数
        image_id = 0  # 保存的图片编号

        while count < self.video_frames:
            ret, image = self.vid.read()
            if not ret:
                break

            # 如果当前帧是1秒整的倍数（第0秒、第1秒、第2秒...）
            if count % self.fps == 0:
                save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, image_id)
                # save_name = os.path.join(frame_save_path, '{:05d}.jpg'.format(image_id))
                cv2.imencode('.jpg', image)[1].tofile(save_name)
                image_id += 1

            count += 1

class AVE_dataset(object):
    def __init__(self, path_to_dataset='/root/autodl-tmp/AVE_Dataset', frame_interval=1, frame_kept_per_second=1):
        self.path_to_video = os.path.join(path_to_dataset, 'AVE')
        self.path_to_audio = os.path.join(path_to_dataset, 'audio_npy_files')
        self.frame_kept_per_second = frame_kept_per_second
        self.sr = 16000

        self.path_to_save = os.path.join(path_to_dataset, 'Image-{:02d}'.format(self.frame_kept_per_second))
        if not os.path.exists(self.path_to_save):
            os.mkdir(self.path_to_save)

        # self.path_to_save_audio = os.path.join(path_to_dataset, 'Audio-{:d}-SE'.format(1004))
        # if not os.path.exists(self.path_to_save_audio):
        #     os.mkdir(self.path_to_save_audio)


        # csv_file = pd.read_csv(os.path.join(path_to_dataset, 'Annotations.txt'))
        with open(os.path.join(path_to_dataset, 'Annotations.txt'), 'r') as f:
            self.file_list = f.readlines()

    def extractImage(self):

        for each_video in self.file_list[1:]:
            print('Precessing {} ...'.format(each_video))
            each_video = each_video.split('&')
            video_dir = os.path.join(self.path_to_video, each_video[1]+'.mp4')
            self.videoReader = videoReader(video_path=video_dir, frame_kept_per_second=self.frame_kept_per_second)

            save_dir = os.path.join(self.path_to_save, each_video[1])
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.videoReader.video2frame_per_second(frame_save_path=save_dir)  # 每个视频最少取10张图片

    def extractImage_SE(self):

        for each_video in self.file_list[1:]:
            print('Precessing {} ...'.format(each_video))
            each_video = each_video.split('&')
            start_t = int(each_video[3])
            end_t = int(each_video[4])
            # print(start_t, end_t)

            video_dir = os.path.join(self.path_to_video, each_video[1]+'.mp4')
            self.videoReader = videoReader(video_path=video_dir, frame_kept_per_second=self.frame_kept_per_second)

            save_dir = os.path.join(self.path_to_save, each_video[1])
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.videoReader.video2frame_update_SE(frame_save_path=save_dir, min_save_frame=10,
                                                   start_t=start_t, end_t=end_t)  # 每个视频最少取10张图片


ave = AVE_dataset()
ave.extractImage()



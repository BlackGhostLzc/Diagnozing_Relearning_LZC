## AVE数据集的预处理
AVE的数据集如下所示：
```
.
├── AVE
├── Annotations.txt
├── ReadMe.txt
├── testSet.txt
├── trainSet.txt
├── valSet.txt
```
其中AVE目录下是原始视频，视频文件命名规则是：
```
Category&VideoID&Quality&StartTime&EndTime.mp4
比如：
Church bell&RUhOCu3LNXM&good&0&10.mp4
```
由于AVE目录下都是视频，并没有把音频模态抽离开，所以需要数据预处理。
1. 从视频中提取音频信息，得到wav文件，wav文件都存储在wave_files目录下。
```py
python mp4_to_wav.py
```
2. 预处理wav文件，然后序列化为tensor形式，所有的文件存储在audio_npy_files目录下。
```
python process_audio.py
```
3. 最后处理视觉模态，从视频中提取帧，图片存储在Image-01-FPS目录下
```
python process_video.py
```

最后，数据预处理结果如下：
```
.
├── Annotations.txt
├── audio_npy_files
├── AVE
├── Image-01-FPS
├── ReadMe.txt
├── testSet.txt
├── trainSet.txt
├── valSet.txt
└── wave_files

```
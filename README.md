# A PyTorch implementation of a YOLO v3 Object Detector

## Contributors
- Le Hoang Phuc
- 

## Requirements
1. Python 3.5
2. OpenCV
3. PyTorch 0.4

Using PyTorch 0.3 will break the detector.

## Running the detector

### On single or multiple images

Clone, and `cd` into the repo directory. The first thing you need to do is to get the weights file
This time around, for v3, authors has supplied a weightsfile only for COCO [here](https://pjreddie.com/media/files/yolov3.weights), and place 

the weights file into your repo directory. Or, if you're on Linux, run below command in terminal.

```
wget https://pjreddie.com/media/files/yolov3.weights 
```
### Speed Accuracy Tradeoff

You can change the resolutions of the input image by the `--reso` flag. The default value is 416. Whatever value you chose, rememeber **it should be a multiple of 32 and greater than 32**. Weird things will happen if you don't. You've been warned. 

```
python detect.py --images imgs --det det --reso 320
```

### On Video
For this, you should run the file, video_demo.py with --video flag specifying the video file. The video file should be in .mp3 format.
```
python video_demo.py --video video.avi
```
or simply run the command below if you already put a video named ***"video.mp3"*** (default value for `--video` is ***"video.mp3"***)
```
python video_demo.py
```
To know more about orther parameters, run file with `-h` flag

### Speeding up Video Inference

To speed video inference, you can try using the video_demo_half.py file instead which does all the inference with 16-bit half precision floats instead of 32-bit float. I haven't seen big improvements, but I attribute that to having an older card (Tesla K80, Kepler arch). If you have one of cards with fast float16 support, try it out, and if possible, benchmark it. 

### On a Camera
This file will take input from your webcam and perform real-time detection. The default image resolution is 160 here, though you can change it with `reso` flag.

```
python cam_demo.py
```

***NOTE:*** 

- You can easily change the code to use different weightsfiles, available at [yolo website](https://pjreddie.com/darknet/yolo/)
- There is tradeoff between FPS and accuracy. You can increase FPS by decreasing input resolution but the accuracy may decrease, and vice versa


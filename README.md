# A PyTorch implementation of a YOLO v3 Object Detector

## Contributors
- Le Hoang Phuc
- Ngo Anh Kiet
- Phan Anh
- Nguyen Thanh Tung
- Kieu Minh Duy

## Requirements
1. Python 3.5
2. OpenCV
3. PyTorch 0.4

***Using PyTorch 0.3 will break the detector*

## Running the detector

### TODO

1.  Clone, and `cd` into the repo directory. The first thing you need to do is to get the weights file, and place the weights file into your repo directory. For v3, authors have published weightfile for COCO ([here](https://pjreddie.com/media/files/yolov3.weights))

    Or, if you are on Linux, run below command in terminal (remember to cd befor run `wget`)

```
wget https://pjreddie.com/media/files/yolov3.weights 
```
2.  Copy or download a video into the repo and name is as ***"video.mp4"***, just to ensure the code runs even when you do not use `--video` flag (this flag will be mentioned later)
### FPS - Accuracy Tradeoff
You can increase FPS by decreasing input resolution but the accuracy may decrease, and vice versa.

You can change the resolution of the input image by the `--reso` flag. The default value is 416. Whatever value you chose, rememeber **it should be a multiple of 32 and greater than 32**.

```
python detect.py --images imgs --det det --reso 320
```

### Run detection on a video
For this, you should run the file, video_demo.py with --video flag specifying the video file. The video file should be in .mp4 format.
```
python video_demo.py --video video.avi
```
or simply run the command below if you have already put a video named ***"video.mp4"*** (default value for `--video` is ***"video.mp4"***)
```
python video_demo.py
```
To know more about orther parameters, run file with `-h` flag.

### Speeding up video inference

To speed video inference, you can try using the **video_demo_half.py** file instead which does all the inference with 16-bit half precision floats instead of 32-bit float. However there is no big improvements on ours hardwares. If you have one of cards with fast float16 support, try it out, and if possible, benchmark it.

### Run detection on a camera
This file will take input from your webcam and perform real-time detection. The default image resolution is 160 here, though you can change it with `reso` flag. But remember about the tradeoff mentioned above.

```
python cam_demo.py
```

***NOTE:*** You can easily change the code to use different weightsfiles, available at [yolo website](https://pjreddie.com/darknet/yolo/)

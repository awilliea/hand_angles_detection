# Hand_angles_detection



### Usage
```python=
# process video
python detect_angles.py --filenames hands_1.mp4 hands_2.mp4 hands_3.mp4 hands_4.mp4 --output_dir Outputs --data_type videos --start_frames 0 0 0 0 --end_frames -1 -1 -1 -1

# process image
* python detect_angles.py --filenames image_1.jpg image_2.jpg image_3.jpg --output_dir Outputs --data_type images
```

### TODO
1. output the max angle and the min angle (each cycle)
2. detect the gradient of angles 

### 第一次討論
* 測量方式：單次測量會有誤差，可以用多次測量再取平均值
* 之後做成 app 的時候，可以設計成標準的測量方式（像是 iphone 在做臉部辨識時的前置動作，讓我們把臉對準某一個輪廓 -> 靜置測量一陣子 -> 轉向測量 等等）
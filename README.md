# Hand_angles_detection
### Related papers
https://drive.google.com/drive/folders/1xxPDvrQ9xO2RgIvJIwXZe_n1rn1aZpIZ?usp=sharing

### Usage
1. Download the entire program file (in the upper right corner of the screen there is a green "code", click on it and there will be a "download zip" below). After downloading, unzip it at your desired location.
2. Change the directory. Right-click on the downloaded file to check its location, open the command line (terminal), and enter the following command. This will change the current location to inside the downloaded project. For example, if you downloaded the file to C:\Users\USER\Desktop and the name of the file you downloaded is "hand_angles_detection", you need to enter in the terminal: cd C:\Users\USER\Desktop\hand_angles_detection.
```bash=
cd location_of_the_downloaded_dir
```
3. Based on the requirements.txt, installing all the packages 
```bash=
pip install -r requirements.txt
```
4. Place the file you wish to analyze into a folder named "data" inside the project directory. If you want to analyze a video, place the file in data/videos; if you want to analyze an image, place the file in data/images.
5. Run the desired program. There are three types to choose from: simple camera test, video analysis, and image file analysis. Choose one. Here, we use video analysis as an example. Copy and paste the code under # analyze video into your command line.
6. Parameters:
    * python version（e.g., C:/Desktop/msys64/bin/python），or it would use the default one
    * --filenames: Replace the xxx.mp4 after --filenames with the name of the video you want to analyze. For instance, if you wish to analyze videos named abc.mp4 and bcd.mp4, then it would change to --filenames abc.mp4 bcd.mp4. Additionally, the video names here must match exactly with what you placed in data/videos.
    * --output_dir: The name after --output_dir indicates where the analyzed files will be saved. You can leave this as it is.
    * --data_type: It depends on the format of the data you want to analyze; you can only input videos or images.
    * --start_frames: It refers to the position in the video where you want to begin the analysis, measured in frames. Each video corresponds to a start frame.
    * --end_frames It refers to the position in the video where you want to end the analysis, measured in frames. Each video corresponds to an end frame. -1 means there's no interruption, and the analysis will run until the video ends.
```python=
# camera test
python camera_test.py

# analyze video
python detect_angles.py --filenames hands_1.mp4 hands_2.mp4 hands_3.mp4 hands_4.mp4 --output_dir Outputs --data_type videos --start_frames 0 0 0 0 --end_frames -1 -1 -1 -1

# analyze image
python detect_angles.py --filenames image_1.jpg image_2.jpg image_3.jpg --output_dir Outputs --data_type images
```

7. Check the results. The results will be stored in the "Outputs" folder, saved as "output_dataname". For instance, if you analyzed a file named abc.mp4, its output results will be in "output_abc". Each output result includes the following data:
   * detected images: Image files marking the hand position, measured in frames. The filename indicates its frame number in the video.
   * parameters.json: Contains various video-related data and the hyperparameters used, such as fps (frames per second), start/end frame, analysis success rate, etc.
   * left/right_hands_coordinates.csv: The 3D positions of each joint (21 points) in every frame for the left/right hand. If not detected, the "Detected" column will display "False".
   * left/right_hands_angles.csv: Angles between the joints of the left/right hand. The column fields correspond to three types of angles calculated for each finger (_1, _2, _3). Specifically, _1 represents the angle from the wrist to the first joint of that finger, and so on. The rows indicate angles corresponding to different frames.
   * left/right_hands_angles_max_min.csv: Maximum and minimum angles between the joints of the left/right hand, the time when the maximum value occurred, and the time when the minimum value occurred. Each column might 

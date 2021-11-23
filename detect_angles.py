import cv2
import math
import numpy as np
import os
import pandas as pd
import mediapipe as mp
from utils.utils import process_lanmark, output_max_min
import argparse
import json

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help='directory contains the images or videos you want to process')
parser.add_argument('--output_dir', default='Outputs',
                    help='directory which you want to output the processed data')
parser.add_argument('--data_type', default="image",
                    help='which type of data you want to process, video or image')
parser.add_argument('--filenames', default=['all'], nargs="*", type=str,
                    help='the name of files you want to process. If you do not assign a specific name of file, it would process all of the files in the data dir')
parser.add_argument('--max_num_hands', default=2, type=int,
                    help='The maximum hands in the video or image')
parser.add_argument('--min_detection_confidence', default=0.5, type=float,
                    help='The prediction confidence of the mediapipe model')
parser.add_argument('--threshold', default=0.2, type=float,
                    help='The threshold for the detection of maximization and minization of the angles')
parser.add_argument('--start_frames', default=[], nargs="*", type=int,
                    help='The starting frames you want to process in a video.')                    
parser.add_argument('--end_frames', default=[], nargs="*", type=int,
                    help='The ending frames you want to process in a video. -1 means you want to process the whole video.')
parser.add_argument('--save_freq',  default=3, type=int,
                    help='The frequency for you to save detected images. e.g., 3 means that you would save one of the three detected images')

                    
class hand_angle_detector:
  def __init__(self,filename, data_type, max_num_hands, min_detection_confidence, threshold, start_frame, end_frame, save_freq, output_dir):
    '''
    filename: str, the name of file which you want to process
    data_type: str, video or image
    max_num_hands: int, default is 2
    min_detection_confidence: int, the confidence fo the detector
    threshold: float, which is between 0 and 1
    start_frame: str,
    end_frame: str,
    output_dir: str, the dir you want to save your outputs
    '''
    self.filename = filename
    self.data_type = data_type
    self.end_frame = end_frame
    self.start_frame = start_frame
    self.output_dir = output_dir
    self.mp_hands = mp.solutions.hands
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.max_num_hands = max_num_hands
    self.min_detection_confidence = min_detection_confidence
    self.threshold = threshold
    self.save_freq = save_freq
    self.image_list = []
    self.new_image_list = []
    self.landmark_label_list = []
    self.detected_rate = []
    self.create_fingle_list()

  # deframe
  def read_video(self):
    '''
    Input:
      video_name: name of the video you want to analyze 
    Output:
      image_list: images deframes from the video
    '''
    if self.end_frame == -1:
        self.end_frame = math.inf
        
    cap = cv2.VideoCapture(self.filename)
    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    self.fps = fps

    image_list = []
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        count += 1
        # print(frame, ret)
        if ret and count >= self.start_frame:
          if len(image_list) < self.end_frame:
              # cv2_imshow(frame)
              image_list.append(frame)
              cv2.waitKey(1)
          else:
              break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    self.image_list = image_list
    self.image_shape = image_list[0].shape
    
  def read_image(self):
    img = cv2.imread(self.filename)
    self.image_list = [img]
    cv2.destroyAllWindows()
  
  # 1 seconds: 3 cycles (6 actions -> 12 ~ 18 images -> 1/3 ~ 1/2 frames)
  # save the analyzed images
  def save_image(self, output_image_dir):

    if not os.path.isdir(output_image_dir):
      os.mkdir(output_image_dir)
    for i,img in enumerate(self.new_image_list):
      file_name = 'frame_{}.jpg'.format(i)
      if i % self.save_freq == 0:
        cv2.imwrite(os.path.join(output_image_dir,file_name), img)
        cv2.waitKey(1)

  # Get the coordinates of each finger point 
  def turn_list_to_df(self):
    column_names = ['Detected']+['point_{}'.format(i) for i in range(21)]
    right_hands_df = pd.DataFrame(columns=column_names)
    left_hands_df = pd.DataFrame(columns=column_names)
    
    for i, landmark_labels in enumerate(self.landmark_label_list):
      if landmark_labels == 'None':
        right_hands_df.loc[i,'Detected'] = False
        left_hands_df.loc[i,'Detected'] = False
      else:
        right = True
        left = True
        for landmark_label in landmark_labels:
          label = landmark_label[0]
          if label == 'Right':
            right = False
            right_hands_df.loc[i,'Detected'] = True
            right_hands_df.loc[i, ['point_{}'.format(i) for i in range(21)]] = landmark_label[1]
          else:
            left = False
            left_hands_df.loc[i,'Detected'] = True
            left_hands_df.loc[i, ['point_{}'.format(i) for i in range(21)]] = landmark_label[1]
        if right:
          right_hands_df.loc[i,'Detected'] = False
        if left:
          left_hands_df.loc[i,'Detected'] = False

    return right_hands_df, left_hands_df

  def compute_angle(self, v1, v2, acute=True):
    # v1 is your firsr vector
    # v2 is your second vector
    # actue= True means that all angles we compute are less than 180 degrees
    angle_ = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        angle = angle_
    else:
        angle =  2 * np.pi - angle_
    return round((180 * angle/np.pi),2)

  def create_fingle_list(self):
    finger_list_point = np.array(['_1','_2','_3']*5)
    finger_list_type = np.array(['thumb']*3+['index_finger']*3+['middle_finger']*3+['ring_finger']*3+['pinky']*3)
    finger_list = []
    for i in range(finger_list_point.shape[0]):
      finger_list.append(finger_list_type[i]+finger_list_point[i])

    self.finger_list = finger_list
  
  def compute_hang_angle_for_each_point(self, hands_df):
  
    angle_list_df = pd.DataFrame(columns=self.finger_list)
    for frame_idx in range(hands_df.shape[0]):
      angle_list = []
      if hands_df.loc[frame_idx,'Detected'] == False:
        angle_list_df.loc[frame_idx] = [math.nan]*15
        continue

      for start_index in range(1,21,4):
        for i in range(3):
          if i == 0:
            v1 = hands_df.loc[frame_idx,'point_{}'.format(start_index+i)] - hands_df.loc[frame_idx,'point_{}'.format(0)]
          else:
            v1 = hands_df.loc[frame_idx,'point_{}'.format(start_index+i)] - hands_df.loc[frame_idx,'point_{}'.format(start_index+i-1)]
          v2 = hands_df.loc[frame_idx,'point_{}'.format(start_index+i+1)] - hands_df.loc[frame_idx,'point_{}'.format(start_index+i)]

          angle = self.compute_angle(v1, v2, True)
          angle_list.append(angle)
      angle_list_df.loc['angles for frame_{}'.format(frame_idx)] = angle_list
    return angle_list_df

  def compute_max_and_min_for_angle_df(self, angles_df):
    max_min_angle_list_df = pd.DataFrame(columns=self.finger_list)
    max_lists, min_lists, max_time_lists, min_time_lists = output_max_min(angles_df, self.fps, self.threshold)
    max_min_angle_list_df.loc['max_angles'] = max_lists
    max_min_angle_list_df.loc['min_angles'] = min_lists
    max_min_angle_list_df.loc['max_times'] = max_time_lists
    max_min_angle_list_df.loc['min_times'] = min_time_lists

    return max_min_angle_list_df

  def detect_hand_images(self):
    new_image_list = []
    landmark_label_list = []
    count = 0
    with self.mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=self.max_num_hands,
        min_detection_confidence=self.min_detection_confidence) as hands:
      for i,image in enumerate(self.image_list):
        # Convert the BGR image to RGB, flip the image around y-axis for correct
        # handedness output and process it with MediaPipe Hands.
        results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

        # Print handedness (left v.s. right hand).
        # print(f'Handedness of picture:')
        # print(results.multi_handedness)

        if not results.multi_hand_landmarks:
          new_image_list.append(image)
          landmark_label_list.append('None')
          count += 1
          continue
        # Draw hand landmarks of each hand.
        # print(f'Hand landmarks of picture:')
        image_hight, image_width, _ = image.shape
        annotated_image = cv2.flip(image.copy(), 1)
        landmark_label = []
        for h, hand_landmarks in enumerate(results.multi_hand_landmarks):
          label = results.multi_handedness[h].classification[0].label
          processed_landmarks = process_lanmark(hand_landmarks.landmark,image_width,image_hight)
          landmark_label.append((label, processed_landmarks))
          # Print index finger tip coordinates.
          # print(
          #     f'Index finger tip coordinate: (',
          #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
          # )
          self.mp_drawing.draw_landmarks(
              annotated_image,
              hand_landmarks,
              self.mp_hands.HAND_CONNECTIONS,
              self.mp_drawing_styles.get_default_hand_landmarks_style(),
              self.mp_drawing_styles.get_default_hand_connections_style())
        landmark_label_list.append(landmark_label)
        new_image_list.append(cv2.flip(annotated_image, 1))
    self.new_image_list = new_image_list
    self.landmark_label_list = landmark_label_list
    self.detected_rate.append(1.0 - count/len(new_image_list))

  def save_hyperparamters(self, dir):

    parameters = {
      'start_frame': self.start_frame,
      'end_frame': self.end_frame,
      'max_num_hands': self.max_num_hands,
      'min_detection_confidence': self.min_detection_confidence,
      'detected_rate': self.detected_rate,
      'fps': self.fps,
      'image_shape': self.image_shape
    }
    with open(os.path.join(dir,'parameters.json'), 'w') as f:
      json.dump(parameters, f)

  def main(self):
    # Check dir
    if not os.path.isdir(self.output_dir):
      os.mkdir(self.output_dir)

    output_file_name =  os.path.basename(self.filename).split('.')[0]
    file_output_dir = os.path.join(self.output_dir,"output_{}".format(output_file_name))

    if not os.path.isdir(file_output_dir):
      os.mkdir(file_output_dir)

    # get raw image_list
    if self.data_type == 'images':
      self.read_image()

    elif self.data_type == 'videos':
      self.read_video()

    # Start detected
    print('Start detected file {}'.format(self.filename))  
    self.detect_hand_images()
    self.save_image(os.path.join(file_output_dir, "detected_images"))
    right_hands_df, left_hands_df = self.turn_list_to_df()
    right_hands_df.to_csv(os.path.join(file_output_dir, 'right_hands_coordinates.csv'))
    left_hands_df.to_csv(os.path.join(file_output_dir, 'left_hands_coordinates.csv'))
    print("Save the coordinates of hands")

    right_hand_angle_df = self.compute_hang_angle_for_each_point(right_hands_df)
    left_hand_angle_df = self.compute_hang_angle_for_each_point(left_hands_df)
    right_hand_angle_df.to_csv(os.path.join(file_output_dir, 'right_hands_angles.csv'))
    left_hand_angle_df.to_csv(os.path.join(file_output_dir, 'left_hands_angles.csv'))
    print("Save the angles of each point of hands")

    right_max_min_angle_list_df = self.compute_max_and_min_for_angle_df(right_hand_angle_df)
    left_max_min_angle_list_df = self.compute_max_and_min_for_angle_df(left_hand_angle_df)
    right_max_min_angle_list_df.to_csv(os.path.join(file_output_dir, 'right_hands_angles_max_min.csv'))
    left_max_min_angle_list_df.to_csv(os.path.join(file_output_dir, 'left_hands_angles_max_min.csv'))
    print("Save the max and min angles of each point of hands")


    self.save_hyperparamters(file_output_dir)
    print("Save hyperparameters")

    print("Complete the detection of file {}".format(self.filename))

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = os.path.join(args.data_dir, args.data_type)
    
    if args.filenames == ['all']:
      filenames = os.listdir(data_dir)
      start_frames = [0 for i in range(len(filenames))]
      end_frames = [-1 for i in range(len(filenames))]
    else:
      filenames = args.filenames
      start_frames = args.start_frames
      end_frames = args.end_frames
    
    if args.data_type == 'videos': 
      assert len(start_frames) == len(end_frames), "the length of start_frames is not the same as the one of end_frames {}, {}".format(start_frames,end_frames)
      assert len(start_frames) == len(filenames), "the length of start_frames is not the same as the one of filenames {}, {}".format(start_frames,filenames)
    else:
      start_frames = [0 for i in range(len(filenames))]
      end_frames = [-1 for i in range(len(filenames))]

    filenames = [os.path.join(data_dir,f) for f in filenames]
    for file_idx, filename in enumerate(filenames):
      start_frame = start_frames[file_idx]
      end_frame = end_frames[file_idx]
      detector = hand_angle_detector(filename, data_type = args.data_type, max_num_hands=args.max_num_hands, \
                                             min_detection_confidence=args.min_detection_confidence, threshold=args.threshold,\
                                             start_frame=start_frame, end_frame=end_frame, save_freq=args.save_freq, output_dir=args.output_dir)
      detector.main()

# python detect_angles.py --filenames hands_1.mp4 hands_2.mp4 hands_3.mp4 hands_4.mp4 --output_dir Outputs --data_type videos --start_frames 0 0 0 0 --end_frames -1 -1 -1 -1
# python detect_angles.py --filenames image_1.jpg image_2.jpg image_3.jpg --output_dir Outputs --data_type images
# python detect_angles.py --filenames 001.mp4 --output_dir Outputs --data_type videos --start_frames 0 --end_frames -1
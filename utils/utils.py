import cv2
import math
import numpy as np
import os
import pandas as pd
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 對座標作轉換處理，原本他給的是該點位在圖片中的比例，這邊直接轉成具體數值
# x 座標表示的是從照片左邊算到該點的距離, y 座標表示的是從該照片上面算到該點的距離, z 座標是與 camera 的距離，越小黎相機躍進
# 圖片的 size (720, 1280)
def process_lanmark(landmarks, width, height):
    processed_landmarks = []
    for landmark in landmarks:
        processed_landmarks.append(np.array([landmark.x*width, landmark.y*height, landmark.z*width]))
    return processed_landmarks

def output_max_min(hand_angle_df, fps, threshold=0.2):
    max_lists = []
    min_lists = []
    max_time_lists = []
    min_time_lists = []

    for i in range(0,15):
        # preprocess
        # print(hand_angle_df.iloc[:,i])
        # print(fps,threshold)
        # input("wait")
        hand_angles = hand_angle_df.iloc[:,i].rolling(fps).mean()
        max_list, min_list = detect_max_min_by_threshold(hand_angles, threshold)
        max_times, min_times = get_time_index(max_list, min_list, hand_angles)
        max_lists.append(max_list)
        min_lists.append(min_list)
        max_time_lists.append(max_times)
        min_time_lists.append(min_times)
    return max_lists, min_lists, max_time_lists, min_time_lists

def detect_max_min_by_threshold(point_angles, threshold=0.5):
    # Remove nan
    # nan_array = np.isnan(point_angles)
    # not_nan_array = ~ nan_array
    # point_angles_true = point_angles[not_nan_array]

    max_list = []
    min_list = []
    min = 360
    max = 0
    start_max = False
    start_min = True

    for i,angle in enumerate(point_angles):
        if np.isnan(angle):
            continue
            
        # update min
        if start_min:
            if angle < min:
                min = angle

            up_threshold = min*(1.0+threshold)
            if angle >= up_threshold:
                start_max = True
                start_min = False
                min_list.append(min)
                min = 360
            elif i == len(point_angles)-1:
                min_list.append(min)
            
        # update max
        if start_max:
            if angle > max:
                max = angle

            down_threshold = max/(1.0+threshold)
            if angle <= down_threshold:
                start_max = False
                start_min = True
                max_list.append(max)
                max = 0
            elif i == len(point_angles)-1:
                max_list.append(max)


    # detect cycle
    return max_list, min_list
    
def get_time_index(max_list, min_list, angles_array):
    max_times = []
    min_times = []

    for m in max_list:
        max_times.append(np.where(angles_array == m)[0][0])

    for m in min_list:
        min_times.append(np.where(angles_array == m)[0][0])

    return np.array(max_times), np.array(min_times) 
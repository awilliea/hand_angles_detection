import cv2
import math
import numpy as np
import os
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt

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
        # hand_angles = round(hand_angle_df.iloc[:,i].rolling(fps).mean(), 1)
        # fillnan first
        hand_angles = hand_angle_df.iloc[:,i].fillna(method='ffill')
        hand_angles = hand_angles.rolling(fps).mean()
        max_list, min_list = detect_max_min_by_threshold(hand_angles, threshold)
        max_times, min_times = get_time_index(max_list, min_list, hand_angles)
        max_list = [np.round(x,1) for x in max_list]
        min_list = [np.round(x,1) for x in min_list]
        # print(max_times)
        # print(min_times)
        # input("")
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
        # print(angle)
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
    # max_times = np.array(max_times)
    # min_times = np.array(min_times)
    return max_times, min_times

def plot_15angles_figure(angles, fps, output_filename, output_smooth_filename):
    plt.figure(figsize=(15,8))
    print(angles)
    for i in range(0,15):
        plt.subplot(5,3,i+1)
        plt.plot(angles.iloc[:,i].values)
    plt.savefig(output_filename)
    plt.close()

    plt.figure(figsize=(15,8))
    angles_fillna = angles.fillna(method='ffill')
    print(angles_fillna)
    for i in range(0,15):
        plt.subplot(5,3,i+1)
        # hand_angles = angles.iloc[:,i].fillna('ffill',inplace=True)
        
        # if hand_angles:
        smooth_angles = angles_fillna.iloc[:,i].rolling(fps).mean()
    
        plt.plot(smooth_angles.values)


    plt.savefig(output_smooth_filename)
    plt.close()


def transform_df_to_doctor_domain(angles_max_min, hand_type='L'):
    df_list = []
    column_names = [hand_type+c for c in ['T','I','M','R','L']]
    # print(angles_max_min.values.shape)
    for row in angles_max_min.values:
        data = []
        for i in range(3):
            # print(i)
            values = []
            for j in range(5):
                index = (2-i)+j*3
                value =row[index]
                values.append(str(value).replace('[','').replace(']',''))
            data.append(values)
        df = pd.DataFrame(columns=column_names,index=['DIPJ','PIPJ','MCPJ'], data=data)
        df_list.append(df)
    return df_list

def save_df_to_doctor_domain(right_hand_angles_max_min, left_hand_angles_max_min, output_filename):
    right_df_list = transform_df_to_doctor_domain(right_hand_angles_max_min, hand_type='R')
    left_df_list = transform_df_to_doctor_domain(left_hand_angles_max_min, hand_type='L')
    name_list = ['max_angles','min_angles','max_angles_framesID','min_angles_framesID']
    with pd.ExcelWriter(output_filename) as writer:  
        for i, df in enumerate(left_df_list):
            all_df = pd.concat([df,right_df_list[i]], axis=1)
            all_df.to_excel(writer, sheet_name=name_list[i])

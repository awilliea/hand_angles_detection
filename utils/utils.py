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



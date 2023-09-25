import cv2
import mediapipe as mp
import numpy as np
import socket
import json

# 初始化 Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 初始化 Unity 連接
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #UDP宣告
dest_addr = ('127.0.0.1', 5714)

def send_pose_data(pose_data):
    json_data = json.dumps(pose_data)
    text = json_data.encode('utf-8')
    udp_socket.sendto(text, dest_addr)

def Pose_Images():
    #使用算法包进行姿态估计时设置的参数
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.8) as pose:
        #打开摄像头
        cap = cv2.VideoCapture(0)
        while(True):
            #读取摄像头图像
            res, image = cap.read() 
            image = cv2.flip(image,1)
            if res is False:
                print('read video error')
                exit(0)
            image.flags.writeable = False
            # Convert the BGR image to RGB before processing.
            # 姿态估计
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                h,w,c = image.shape
                # 提取關鍵點座標
                pose_landmarks = results.pose_landmarks.landmark
                # 將座標轉換為列表
                pose_landmarks_list = []
                for landmark in pose_landmarks:
                    pose_landmarks_list.append([landmark.x, landmark.y, landmark.z])
                # 傳送姿態資料到 Unity
                send_pose_data(pose_landmarks_list)
                print(pose_landmarks_list)
                print("1",len(pose_landmarks))
            cv2.imshow('image', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):       # 按q退出
                break
        cap.release()

if __name__ == '__main__':
    Pose_Images()

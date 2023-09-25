
import cv2
from deepface import DeepFace
import numpy as np

def emotionscore():
    img = cv2.imread("E:\\Unity\\IEEEEXE_low\\pyexe\\dist\\te.png")     # 讀取圖片
    try:
        
        analyze = DeepFace.analyze(img, actions=['emotion'] )  # 辨識圖片人臉資訊，取出情緒資訊
        emo = analyze[0]['dominant_emotion']
        if emo == 'happy':
            return 100
        elif emo == 'neutral':
            return 90
        else :
            return 80
    except:
        return 80


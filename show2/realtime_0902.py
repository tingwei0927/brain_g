import time
import cv2
import mediapipe as mp
import csv
import os
import sys
import for_score0902
import process_file
import emotion
import random


try:
    mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
    mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
    mp_hands = mp.solutions.hands                    # mediapipe 偵測手掌方法

    set1_1 = frozenset({(2,3),(3,4)})
    set1_2 = frozenset({(6,7),(7,8)})
    set1_3 = frozenset({(10,11),(11,12)})
    set1_4 = frozenset({(4,8)})
    set1_5 = frozenset({(4,12)})

    r_finger = []
    l_finger = []
    tmp = []
    epod = 0
    epod2 =0
    hand_dis = []



    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    m=0
    max = 0
    frame_count = 0
    video_name = "./09/09.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, int(cap.get(cv2.CAP_PROP_FPS))/2, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    allframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a123=""
    # mediapipe 啟用偵測手掌
    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            if a123 == "":
                a123 = input()
                start123 = time.time()
            joint_color = (212,199,129)  # 關節點的顏色
            line_color = (0, 0, 255)   # 骨架連接線的顏色
            joint_radius = 3         # 關節點的半徑
            line_thickness = 2         # 骨架連接線的粗細
            
    #         img = cv2.resize(img,(1500,1000))
            img = img[40:280, 160:480]

            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
            image_height, image_width, _ = img.shape
            # img = img[300:image_height, 0:image_width]
            # image_height, image_width, _ = img.shape
            img = cv2.flip(img,1)
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
            results = hands.process(img2)                 # 偵測手掌
            
            
            set2 = []

            

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1:
                frame_count = frame_count+1
                now_time = time.time()-start123
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    q123 = ((hand_landmarks.landmark[5].x*image_width-hand_landmarks.landmark[8].x*image_width)**2+(hand_landmarks.landmark[5].y*image_height-hand_landmarks.landmark[8].y*image_height)**2)**0.5
                    if q123 > max:
                        max = q123
                        
                    set2.append(hand_landmarks.landmark[2])
                    set2.append(hand_landmarks.landmark[3])
                    set2.append(hand_landmarks.landmark[4])
                    set2.append(hand_landmarks.landmark[6])
                    set2.append(hand_landmarks.landmark[7])   
                    set2.append(hand_landmarks.landmark[8])
                    for connection in set1_1:
                        start_index = connection[0]
                        end_index = connection[1]
                        start_point = (int(landmarks[start_index].x * image_width), int(landmarks[start_index].y * image_height))
                        end_point = (int(landmarks[end_index].x * image_width), int(landmarks[end_index].y * image_height))
                        cv2.line(img, start_point, end_point, (255, 255, 255), line_thickness)
                    for connection in set1_2:
                        start_index = connection[0]
                        end_index = connection[1]
                        start_point = (int(landmarks[start_index].x * image_width), int(landmarks[start_index].y * image_height))
                        end_point = (int(landmarks[end_index].x * image_width), int(landmarks[end_index].y * image_height))
                        cv2.line(img, start_point, end_point, (255, 255, 255), line_thickness)
                    for landmark in set2:
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_height)
                        cv2.circle(img, (x, y), joint_radius, joint_color, -1)



                    if epod == 0:
                        tmp.append(landmarks[0])
                        tmp.append(landmarks[4])
                        tmp.append(landmarks[8])
                        epod = 1
                    else :
                        if tmp[0].x > landmarks[0].x : #tmp是右手 先測右手
                            n = ((tmp[2].x - landmarks[4].x)**2 + (tmp[2].y - landmarks[4].y)**2)**0.5
                            r_finger.append(n)
                            n = ((tmp[1].x - landmarks[8].x)**2 + (tmp[2].y - landmarks[8].y)**2)**0.5
                            l_finger.append(n)
                            hand_dis.append(tmp[2].y - landmarks[8].y)
                        
                        else:
                            n = ((tmp[2].x - landmarks[4].x)**2 + (tmp[2].y - landmarks[4].y)**2)**0.5
                            l_finger.append(n)
                            n = ((tmp[1].x - landmarks[8].x)**2 + (tmp[2].y - landmarks[8].y)**2)**0.5
                            r_finger.append(n)
                            hand_dis.append(landmarks[8].y - tmp[2].y )
                        tmp = []    
                        epod = 0
                        m += 1
                        end_time = time.time()
                    
                        
                        
                    



    #         if results.multi_hand_landmarks:
    #             for hand_landmarks in results.multi_hand_landmarks:
    #                 # 將節點和骨架繪製到影像中

    #                 mp_drawing.draw_landmarks(
    #                     img,
    #                     hand_landmarks,
    #                     set1,
    #                     mp_drawing_styles.get_default_hand_landmarks_style(),
    #                     mp_drawing_styles.get_default_hand_connections_style())
    #         print(hand_landmarks[0])

    #         print(mp_hands.HAND_CONNECTIONS)

            cv2.imshow('abc123', img)
            out.write(img)
            if cv2.waitKey(5) == ord('q'):
                break
            if (time.time()-start123)>30:
                break    # 按下 q 鍵停止
    elapsed_time=round(frame_count / fps,2)
    #elapsed_time = 20

    output_folder = "./09/test"
    vvv = "09"
    os.makedirs(output_folder, exist_ok=True)
    csv_name = "./"+vvv+"/test/"+ vvv
    path = csv_name + ".csv"
    with open(path, 'w', newline='') as csvfile:


        writer = csv.writer(csvfile)
        for a in range(m):
    #         if tmp1234[a]/max <1.5:
                writer.writerow([a,(r_finger[a]/max),(l_finger[a]/max),(hand_dis[a]/max),(elapsed_time)])  # 将浮点数转换为列表



    cap.release()
    out.release()
    cv2.destroyAllWindows()
    #print(max)
    stability, accuracy, smooth = for_score0902.process(path)
    # print(stability)
    # print(accuracy)
    # print(smooth)




    process_file.removefile("./09/test")

    process_file.sort_and_rename_files("./save/09",video_name)
except:
    print(f"stb:73")
    print(f"cro:65")
    print(f"flu:68")
    emoscore = emotion.emotionscore()*0.7
    emoscore = int(emoscore + 65*0.3)

    print(f"emo:{emoscore}")
#方形
import time
import cv2
import mediapipe as mp
import csv
import os
import for_score09

def for_video09(video):
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


    vvv = video
    path = "./"+vvv+"/"+vvv+".mp4"
    cap = cv2.VideoCapture(path)
    # cap = cv2.VideoCapture(0)
    m=0
    max = 0

    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    allframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
            joint_color = (0, 255, 0)  # 關節點的顏色
            line_color = (0, 0, 255)   # 骨架連接線的顏色
            joint_radius = 5         # 關節點的半徑
            line_thickness = 2         # 骨架連接線的粗細
            
    #         img = cv2.resize(img,(1500,1000))
            image_height, image_width, _ = img.shape
            # img = img[300:image_height, 0:image_width]
            # image_height, image_width, _ = img.shape
            img = cv2.flip(img,1)
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
            results = hands.process(img2)                 # 偵測手掌
            
            
            set2 = []

            

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    q123 = ((hand_landmarks.landmark[5].x*image_width-hand_landmarks.landmark[8].x*image_width)**2+(hand_landmarks.landmark[5].y*image_height-hand_landmarks.landmark[8].y*image_height)**2)**0.5
                    if q123 > max:
                        max = q123
                        
                        
                    

                    

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
                            cv2.line(img, (int(tmp[2].x*image_width),int(tmp[2].y*image_height)), (int(landmarks[4].x*image_width),int(landmarks[4].y*image_height)), (255, 0, 0), line_thickness)
                            cv2.line(img, (int(tmp[1].x*image_width),int(tmp[1].y*image_height)), (int(landmarks[8].x*image_width),int(landmarks[8].y*image_height)), (255, 255, 0), line_thickness)
                        else:
                            n = ((tmp[2].x - landmarks[4].x)**2 + (tmp[2].y - landmarks[4].y)**2)**0.5
                            l_finger.append(n)
                            n = ((tmp[1].x - landmarks[8].x)**2 + (tmp[2].y - landmarks[8].y)**2)**0.5
                            r_finger.append(n)
                            hand_dis.append(landmarks[8].y - tmp[2].y )
                            cv2.line(img, (int(tmp[1].x*image_width),int(tmp[1].y*image_height)), (int(landmarks[8].x*image_width),int(landmarks[8].y*image_height)), (255, 0, 0), line_thickness)
                            cv2.line(img, (int(tmp[2].x*image_width),int(tmp[2].y*image_height)), (int(landmarks[4].x*image_width),int(landmarks[4].y*image_height)), (255, 255, 0), line_thickness)
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
            #cv2.imshow('oxxostudio', img)
            # if cv2.waitKey(5) == ord('q'):
            #     break    # 按下 q 鍵停止
    elapsed_time=round(allframe / fps,2)

    output_folder = "./"+vvv+'/test'
    os.makedirs(output_folder, exist_ok=True)
    csv_name = "./"+vvv+"/test/"+ vvv
    path = csv_name + ".csv"
    with open(path, 'w', newline='') as csvfile:


        writer = csv.writer(csvfile)
        for a in range(m):
    #         if tmp1234[a]/max <1.5:
                writer.writerow([a,(r_finger[a]/max),(l_finger[a]/max),(hand_dis[a]/max),(elapsed_time)])  # 将浮点数转换为列表



   # print(max)
    stability, accuracy, smooth = for_score09.process(path)


    cap.release()
    cv2.destroyAllWindows()

    return stability, accuracy, smooth

#for_video09("09")
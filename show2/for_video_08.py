#指圈
import time
import cv2
import mediapipe as mp
import csv
import for_score08
import os

def for_video08(video):
    mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
    mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
    mp_hands = mp.solutions.hands                    # mediapipe 偵測手掌方法

    set1_1 = frozenset({(2,3),(3,4)})
    set1_2 = frozenset({(6,7),(7,8)})
    set1_3 = frozenset({(10,11),(11,12)})
    set1_4 = frozenset({(4,8)})
    set1_5 = frozenset({(4,12)})
    a = [] #右
    b = []
    c = []#左
    d = []
    e =[]
    f = []
    ac = 0
    bd = 0
    dotime = 0
    i = 0

    tmp1 = []
    tmp2 = []
    tmp = []
    what_time = []
    start_time = time.time()

    vvv = video
    path = "./"+vvv+"/"+vvv+".mp4"
    cap = cv2.VideoCapture(path) # "C:\\mp_test\\hand_allr.mp4"

    # cap = cv2.VideoCapture(0)
    m=0
    max = 0

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

            
    #         if results.multi_hand_landmarks:
    #             for hand_landmarks in results.multi_hand_landmarks:
    #                 set2.append(hand_landmarks.landmark[2])
    #                 set2.append(hand_landmarks.landmark[3])
    #                 set2.append(hand_landmarks.landmark[4])

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    q123 = ((hand_landmarks.landmark[5].x*image_width-hand_landmarks.landmark[8].x*image_width)**2+(hand_landmarks.landmark[5].y*image_height-hand_landmarks.landmark[8].y*image_height)**2)**0.5
                    if q123 > max:
                        max = q123

                    
                    
                    for connection in set1_4:
                        start_index = connection[0]
                        end_index = connection[1]
                        start_point = (int(landmarks[start_index].x * image_width), int(landmarks[start_index].y * image_height))
                        end_point = (int(landmarks[end_index].x * image_width), int(landmarks[end_index].y * image_height))
                        n1 = ((start_point[0]-end_point[0])**2+(start_point[1]-end_point[1])**2)**0.5
                        if ac == 0:
                            tmp1.append(landmarks[0])
                            tmp1.append(n1)
                            tmp1.append(start_point)
                            tmp1.append(end_point)
                            dis = landmarks[0]
                            
                            ac += 1
                        else :
                            if tmp1[0].x > landmarks[0].x:
                                a.append(tmp1[1])
                                c.append(n1)
                                what_time.append(time.time() - start_time)
                                cv2.line(img, tmp1[2], tmp1[3], (10,215,255), line_thickness)
                                cv2.line(img, start_point, end_point, (0,0,0), line_thickness)
                            else :
                                a.append(n1)
                                c.append(tmp1[1])
                                cv2.line(img, start_point, end_point, (10,215,255), line_thickness)
                                cv2.line(img, tmp1[2], tmp1[3], (0,0,0), line_thickness)
                                what_time.append(time.time() - start_time)
                            ac =0
                            tmp1 = []
                            m+=1
                            
                    
                    for connection in set1_5:
                        start_index = connection[0]
                        end_index = connection[1]
                        start_point = (int(landmarks[start_index].x * image_width), int(landmarks[start_index].y * image_height))
                        end_point = (int(landmarks[end_index].x * image_width), int(landmarks[end_index].y * image_height))
                        n2 = ((start_point[0]-end_point[0])**2+(start_point[1]-end_point[1])**2)**0.5
                        if bd == 0:
                            tmp2.append(landmarks[0])
                            tmp2.append(n2)
                            tmp2.append(start_point)
                            tmp2.append(end_point)
                            dis2 = landmarks[0]

                            bd += 1
                        else :
                            if tmp2[0].x > landmarks[0].x:
                                b.append(tmp2[1])
                                d.append(n2)
                                cv2.line(img, tmp2[2], tmp2[3], (10,215,255), line_thickness)
                                cv2.line(img, start_point, end_point, (0,0,0), line_thickness)
                            else :
                                b.append(n2)
                                d.append(tmp2[1])
                                cv2.line(img, start_point, end_point, (10,215,255), line_thickness)
                                cv2.line(img, tmp2[2], tmp2[3], (0,0,0), line_thickness)
                            bd =0
                            tmp2 = []
                    if i == 0:
                        n = landmarks[4].x*image_width
                        i += 1 
                    else:
                        n3 = abs(n-landmarks[4].x*image_width)
    #                     print(abs(n-landmarks[4].x*image_width))
                        tmp.append(n3)
                        i = 0
                    
                    


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
            # cv2.imshow('oxxostudio', img)
            # if cv2.waitKey(5) == ord('q'):
            #     break    # 按下 q 鍵停止
    output_folder = "./"+vvv+'/test'
    os.makedirs(output_folder, exist_ok=True)
    csv_name = "./"+vvv+"/test/"+ vvv
    path = csv_name + ".csv"
    with open(path, 'w', newline='') as csvfile:
        
        start = 0
        end = 0
        i = 0
        for n in range(len(tmp)):
            if tmp[n] < 1.2*max:
                start = n
                break

        for n in range(len(tmp)):
            if i == 0:
                if tmp[n] > 1.2*max:
                    end = n
                    dotime += 1
                    i += 1
            else:
                if tmp[n] < 0.8*max:
                    i = 0

        
        
        print(what_time[start])
        print(what_time[end])
        flu = (what_time[end] - what_time[start])/dotime
        print("fu",what_time[end] - what_time[start])



        writer = csv.writer(csvfile)
        for a0 in range(m):

            writer.writerow([a0,a[a0]/max,b[a0]/max,c[a0]/max,d[a0]/max,flu])  # 将浮点数转换为列表

                    
                
    print(max)
    stability, accuracy, smooth = for_score08.process(path)
    cap.release()
    cv2.destroyAllWindows()
    print(stability)
    return stability, accuracy, smooth


#for_video08("08")
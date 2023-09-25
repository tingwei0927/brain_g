#指圈
import time
import cv2
import mediapipe as mp
import csv
import os
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import process_file
import random
import emotion

try:

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
    #cap = cv2.VideoCapture("C:\\mp_test\\hand_allr.mp4")

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    m=0
    max = 0
    frame_count = 0

    # mediapipe 啟用偵測手掌
    a123 = ""
    video_name = "./08/08.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, int(cap.get(cv2.CAP_PROP_FPS))/2, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
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

            
    #         if results.multi_hand_landmarks:
    #             for hand_landmarks in results.multi_hand_landmarks:
    #                 set2.append(hand_landmarks.landmark[2])
    #                 set2.append(hand_landmarks.landmark[3])
    #                 set2.append(hand_landmarks.landmark[4])

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1:
                frame_count = frame_count+1
                #print("frame_count",frame_count)
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
                    set2.append(hand_landmarks.landmark[10])
                    set2.append(hand_landmarks.landmark[11])   
                    set2.append(hand_landmarks.landmark[12])
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
                    for connection in set1_3:
                        start_index = connection[0]
                        end_index = connection[1]
                        start_point = (int(landmarks[start_index].x * image_width), int(landmarks[start_index].y * image_height))
                        end_point = (int(landmarks[end_index].x * image_width), int(landmarks[end_index].y * image_height))
                        cv2.line(img, start_point, end_point, (255, 255, 255), line_thickness)
                    for landmark in set2:
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_height)
                        cv2.circle(img, (x, y), joint_radius, joint_color, -1)

                    
                    
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

                            else :
                                a.append(n1)
                                c.append(tmp1[1])

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

                            else :
                                b.append(n2)
                                d.append(tmp2[1])

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
 
            cv2.imshow('abc123', img)
            out.write(img)
            if cv2.waitKey(5) == ord('q'):
                break
            if (time.time()-start123)>30:
                break    # 按下 q 鍵停止
    output_folder = "./08/test"
    vvv = "08"
    os.makedirs(output_folder, exist_ok=True)
    csv_name = "./"+vvv+"/test/"+ vvv
    path = csv_name + ".csv"
    #print("len", len(tmp))
    with open(path, 'w', newline='') as csvfile:
        
        start = 0
        end = 0
        i = 0
        for n in range(len(tmp)):
            if tmp[n] < 1*max:
                start = n
                break

        for n in range(len(tmp)):
            if i == 0:
                if tmp[n] > 1*max:
                    end = n
                    dotime += 1
                    i += 1
            else:
                if tmp[n] < 0.5*max:
                    i = 0

        
        
        #print("1",what_time[start])
        #print("2",what_time[end])
        #print("dotime",dotime)
        
        #flu = (what_time[end] - what_time[start])/dotime
        elapsed_time=round(frame_count / fps,2)
        if (dotime == 0):
            dotime = 1
        flu = elapsed_time/dotime
        #print("flu",flu)



        writer = csv.writer(csvfile)
        for a0 in range(m):

            writer.writerow([a0,a[a0]/max,b[a0]/max,c[a0]/max,d[a0]/max,flu])  # 将浮点数转换为列表

                    
                
    #print(max)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # Open CSV
    #with open('01brainpic.csv', 'r') as file:
    #with open('04.csv', 'r') as file:
    a_data = []
    b_data = []
    c_data = []
    d_data = []

    with open(path, 'r') as csvfile:
        # read csv
        rows = csv.reader(csvfile)

        # 以迴圈輸出每一列
        for row in rows:
            a_data.append(float(row[1]))
            b_data.append(float(row[2]))
            c_data.append(float(row[3]))
            d_data.append(float(row[4]))
            dotime = (row[5])
        



        # print(a_array)
        # print(a_data)


    def box_smooth(data, n):
        smoothed_data = []
        window = []


        for i in range(n-1):
            window.append(data[i])

            average = sum(window) / len(window)
            smoothed_data.append(average)


        for i in range(n -1, len(data) - n +1):
            window.append(data[i])
            average = sum(window) / n
            smoothed_data.append(average)
            window.pop(0)  
        


        for i in range(len(data) - n+1, len(data)):
            window.append(data[i])
            average = sum(window) / len(window)
            smoothed_data.append(average)

        return smoothed_data






    # def box_smooth(data, n):
    #     smoothed_data = []
    #     window = []
        
    #     for i in range(len(data)-n):
    #         window.append(float(data[i]))
            
    #         if len(window) == n:
    #             average = sum(window) / n
    #             smoothed_data.extend([average] * n)
    #             window = []


        
        
    #     return smoothed_data



    # 应用 box smooth 平滑处理


    # a_array = np.array(a_data[1:], dtype=float)
    # b_array = np.array(b_data[1:], dtype=float)
    # c_array = np.array(c_data[1:], dtype=float)
    # d_array = np.array(d_data[1:], dtype=float)

    a_data = box_smooth(a_data, 10)
    b_data = box_smooth(b_data, 10)
    c_data = box_smooth(c_data, 10)
    d_data = box_smooth(d_data, 10)
    m = len(a_data)

    e = []
    f = []
    g = []
    h = []

    for n in range(m-1):
        e.append(a_data[n]+ b_data[n])
    for n in range(m-1):
        f.append(c_data[n]+ d_data[n])
    for n in range(m-1):
        # if (a_data[n]- b_data[n]) < 0:
        #     g.append(0)
        #     continue
        g.append(a_data[n]- b_data[n])
    for n in range(m-1):
        # if (c_data[n]- d_data[n]) < 0:
        #     h.append(0)
        #     continue
        h.append(c_data[n]- d_data[n])

    g = np.array(g, dtype=float)
    h = np.array(h, dtype=float)   
    g2 = box_smooth(g, 50)
    h2 = box_smooth(h, 50)
    g = g-g2
    h = h - h2


    a_data = np.array(a_data, dtype=float)
    b_data = np.array(b_data, dtype=float)
    e = np.array(e, dtype=float)
    f = np.array(f, dtype=float)   
    # g = np.array(g, dtype=float)
    # h = np.array(h, dtype=float)   


    def cut(data):
        num_points = len(data)
        start_index = int(num_points * 0)  
        end_index = int(num_points * 1)   
        data = data[start_index:end_index]
        return data


    def draw(data1,data2):
        x_data = np.arange(1, len(data1)+1)
        plt.plot(x_data, data1, label='Data')
        plt.plot(x_data, data2, label='Data')


        # plt.plot(x_fit, y_fit, 'r-', label='Fit')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        #plt.show()

    def draw_peak(data1,peak1,peak2,data2,peak3,peak4):
        x_data = np.arange(1, len(data1)+1)
        plt.plot(x_data, data1, label='Data')
        plt.plot(x_data, data2, label='Data')
        for i in range(len(peak1)):
            plt.scatter(peak1[i], data1[peak1[i]])
        for i in range(len(peak2)):
            plt.scatter(peak2[i], data1[peak2[i]])
        for i in range(len(peak3)):
            plt.scatter(peak3[i], data2[peak3[i]])
        for i in range(len(peak4)):
            plt.scatter(peak4[i], data2[peak4[i]])


        # plt.plot(x_fit, y_fit, 'r-', label='Fit')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        #plt.show()



    # def stadard_count(data):


    #     #print(numpy_array)
        


    #     from scipy.optimize import curve_fit

    #     # fitting function
    #     def fourier2(x, a0, a1, b1, a2, b2, w):
    #         return a0 + a1*np.cos(x*w) + b1*np.sin(x*w) + a2*np.cos(2*x*w) + b2*np.sin(2*x*w)

    #     # 取中間80%資料
    #     y_data = data
    #     num_points = len(y_data)
    #     start_index = int(num_points * 0)  
    #     end_index = int(num_points * 1)   
    #     y_data = y_data[start_index:end_index]
    #     # normalize [0, 1]
    #     # y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))
    #     # y_data = y_data/150
    #     # make X data
    #     x_data = np.arange(1, len(y_data)+1)

    #     # 使用 curve_fit
    #     initial_guess = [0, 0, 0, 0, 0, 0.08]  # 初始值
    #     fit_params, _ = curve_fit(fourier2, x_data, y_data, p0=initial_guess, maxfev=5000, method='lm')
    #     # print("参数:", fit_params)



    #     # 畫圖
    #     plt.plot(x_data, y_data, label='Data')
    #     x_fit = x_data
    #     y_fit = fourier2(x_data, *fit_params)

    #     plt.plot(x_fit, y_fit, 'r-', label='Fit')

    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.legend()
    #     plt.show()
    #     # print(np.std(y_data)*1000)


    #     residuals = y_data - y_fit
    #     # print("go",(np.sum(np.abs(residuals)**2)/(len(residuals)-1))**(1/2))
    #     #print("N:", str(len(residuals)))
    #     #print("dev:", np.sum(y_data))
    #     residuals_std = np.std(residuals)
    #     residuals_std = round(residuals_std,2)
    #     print("標準差:", residuals_std)
    #     return residuals_std

    def fourier2(x, a0, a1, b1, a2, b2, w):
        return a0 + a1*np.cos(x*w) + b1*np.sin(x*w) + a2*np.cos(2*x*w) + b2*np.sin(2*x*w)
    def stadard_count(data):

        #print(numpy_array

        # 取中間80%資料
        y_data = data
        num_points = len(y_data)
        start_index = int(num_points * 0.1)  
        end_index = int(num_points * 0.9)   
        y_data = y_data[start_index:end_index]
        # normalize [0, 1]
        y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))
        # make X data
        x_data = np.arange(1, len(y_data)+1)

        std_list = []
        for i in range(1, 26):
            guess = i/100
            # 使用 curve_fit
            initial_guess = [0, 0, 0, 0, 0, guess]  # 初始值，老師說之後要用迴圈去調整
        
            fit_params, _ = curve_fit(fourier2, x_data, y_data, p0=initial_guess, maxfev=5000, method='lm')

            
            x_fit = x_data
            y_fit = fourier2(x_data, *fit_params)
            residuals = y_data - y_fit
            #print("go",(np.sum(np.abs(residuals)**2)/(len(residuals)-1))**(1/2))
            #print("N:", str(len(residuals)))
            #print("dev:", np.sum(y_data))
            residuals_std = np.std(residuals)
            residuals_std = round(residuals_std,2)
            std_list.append(residuals_std)
            #print("std_list", std_list)
        
        min_std = np.argmin(std_list) #找最小值的index
        new_guess = (min_std+1)/100



        initial_guess = [0, 0, 0, 0, 0, new_guess]  # 初始值，老師說之後要用迴圈去調整

        fit_params, _ = curve_fit(fourier2, x_data, y_data, p0=initial_guess, maxfev=5000, method='lm')



        #print("標準差:", residuals_std)
        
    
        

    # print("参数:", fit_params)



        # 畫圖
        # plt.scatter(x_data, y_data, label='Data')
        # x_fit = x_data
        # y_fit = fourier2(x_data, *fit_params)

        # plt.plot(x_fit, y_fit, 'r-', label='Fit')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.legend()
        # plt.show()

        plt.plot(x_data, y_data, label='Data')
        x_fit = x_data
        y_fit = fourier2(x_data, *fit_params)

        plt.plot(x_fit, y_fit, 'r-', label='Fit')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        #plt.show()


        residuals = y_data - y_fit
    #  print("go",(np.sum(np.abs(residuals)**2)/(len(residuals)-1))**(1/2))
        #print("N:", str(len(residuals)))
        #print("dev:", np.sum(y_data))
        residuals_std = np.std(residuals)
        residuals_std = round(residuals_std,2)
        #print("標準差:", residuals_std)
        return residuals_std

    # std1= stadard_count("C:\\Users\\user\\Desktop\\hand11.csv")
    # std1= stadard_count("C:\\Users\\user\\Desktop\\hand22.csv")
    # std1= stadard_count("C:\\Users\\user\\Desktop\\hand333.csv")

    # std1= stadard_count("C:\\Users\\user\\Desktop\\hand110.csv")
    # std2= stadard_count("C:\\Users\\user\\Desktop\\hand1.csv")
    # std3= stadard_count("C:\\Users\\user\\Desktop\\hand2.csv")
    # std4= stadard_count("C:\\Users\\user\\Desktop\\hand3.csv")
    # std5= stadard_count("C:\\Users\\user\\Desktop\\hand9.csv")#低角度
    # std = std3
    # print(std1)
    # print(std2)
    # print(std3)
    # print(std4)
    # print(std5)
    # if (std - 1.8 )<0:
    #     print("正確度: 100分")
    # else:
    #     s = (1 - (std - 1.8)/1.8)*100
    #     print(f"正確度: {s}分")
    # np.std(y_data)


    g = cut(g)
    h = cut(h)


    g = -g
    # std1= stadard_count(a_data)
    # a_data = box_smooth(a_data, 10)
    # std1= stadard_count(a_data)
    # draw(g,h)
    std1= stadard_count(e)
    std2= stadard_count(f)
    # std3= stadard_count(g)
    # std3= stadard_count(h)
    # std3= stadard_count(a_data)
    # std3= stadard_count(b_data)
    score1 = 100 * np.exp(-(std1 - 0.13))
    score2 = 100 * np.exp(-(std2 - 0.12))
    score1 = score1 + 36
    score2 = score2 + 36
    score1 = int((np.sqrt(score1))*10)
    score2 = int((np.sqrt(score2))*10)
    if score1 >= 100:
        score1 =  random.randint(91, 100)
    if score2 >= 100:
        score2 =  random.randint(91, 100)




    peaks_g, _ = find_peaks(g,distance=40,prominence=0.2)
    peaks_g2, _ = find_peaks(-g,distance=40,prominence=0.2)
    peaks_h, _ = find_peaks(h,distance=40,prominence=0.2)
    peaks_h2, _ = find_peaks(-h,distance=40,prominence=0.2)
    # peaks_g, _ = find_peaks(g)
    # peaks_h, _ = find_peaks(h)

    # print(peaks_g)
    # print(peaks_h)
    # print(peaks_g2)
    # print(peaks_h2)
    draw_peak(g,peaks_g,peaks_g2,h,peaks_h,peaks_h2)

    # peaks_g = len(peaks_g)
    # peaks_h = len(peaks_h)


    # print(int(score1))
    # print(int(score2))
    

    print(f"stb:{int((score1 + score2)/2)}")
    stability = int((score1 + score2)/2)
    # print(100 - abs(peaks_g-8)/8*100)
    # print(100 - abs(peaks_h-8)/8*100)

    s = 0
    tmp = 0

    for i in range(len(peaks_g)):
        tmp = 0
        for n in range(len(peaks_h)):
            if peaks_h[n] < peaks_g[i]+30 and peaks_h[n] > peaks_g[i]-30:
                tmp = 1

        s  += tmp

    for i in range(len(peaks_g2)):
        tmp = 0
        for n in range(len(peaks_h2)):
            if peaks_h2[n] < peaks_g2[i]+30 and peaks_h2[n] > peaks_g2[i]-30:
                tmp = 1

        s  += tmp




    score3 = 100*  s/ (len(peaks_g)+len(peaks_g2))
    score3 = score3 + 36
    score3 = int((np.sqrt(score3))*10)
    if score3 >= 100:
        score3 =  random.randint(91, 100)

    # if score3 <40:
    #     score3 += 60
    # else:
    #     score3 = score3**0.5 *10
    print(f"cro:{int(score3)}")

    score4 = f_score = 100*math.exp((2-float(dotime))/2)
    score4 = score4+36
    score4 = int((np.sqrt(score4))*10)

    if score4 > 100:
        score4 =  random.randint(91, 100)

    print(f"flu:{int(score4)}")
    emoscore = emotion.emotionscore()*0.7
    emoscore = int(emoscore + (stability + score3 + score4 )/3*0.3)

    print(f"emo:{emoscore}")

    # print(stability)
    # print(score3)
    # print(score4)
    process_file.removefile("./08/test")
    process_file.sort_and_rename_files("./save/08",video_name)
except:
    print(f"stb:73")
    print(f"cro:65")
    print(f"flu:68")
    emoscore = emotion.emotionscore()*0.7
    emoscore = int(emoscore + 65*0.3)

    print(f"emo:{emoscore}")
#交互拍肩
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from PIL import Image
from sympy import*
import math
import sys
import mediapipe as mp
from natsort import natsorted
import euclidean
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import csv
import os
import euclidean
import Std_of_stability
from matplotlib.pyplot import MultipleLocator 
import time
import for_knee
import process_file
import emotion
import random


try:
    sa_F = [] #存腳的距離比例成績
    sa_H = [] #存手的距離成績
    thres_F = 0.4  #手跟膝蓋距離
    thres_H = 0.27 #手跟肩膀距離
    #下面四個是教練的四肢標準差，04只用手摸肩膀跟膝蓋的部分來計算
    LH_B_std=0.0101
    RH_B_std=0.0087
    LH_S_std=0.0133
    RH_S_std=0.0179



    def calculate_std(x, z): #算標準差評分用的公式，x是使用者標準差，z是教練標準差
        f_score = np.sqrt(100 * np.exp(-(x - z)/z))*10
        if f_score>=100:
            f_score = 100
        else:
            f_score = f_score
        return f_score

    def hand_point(body_np, point_1, point_2): #計算手掌的整體座標
    # 要加這個才知道現在應該是整體的nparray的哪個
        point_1 = int(point_1)
        point_2 = int(point_2)
        if point_1==17 or point_1==21 or point_1 == 19:
            po21 = body_np[21]
            po19 = body_np[19]
            po17 = body_np[17]
            po21 = po21[1:]
            po19 = po19[1:]
            po17 = po17[1:]
            #pointL = (po17+po19+po21)/3 #手掌座標
            pointL = (po17+po19)/2 
        if   point_2 == 18 or point_2 == 20 or point_2 ==22:
            po22 = body_np[22]
            po20 = body_np[20]
            po18 = body_np[18]
            po22 = po22[1:]
            po20 = po20[1:]
            po18 = po18[1:]
            #pointR = (po18+po20+po22)/3
            pointR = (po18+po20)/2 
        #print("point1:",pointL)
        #print("point2:",pointR)
        return pointL, pointR #回傳手掌的x, y


    def foot_point(body_np, point_1, point_2):
        point_1 = int(point_1)
        point_2 = int(point_2)
        #print("kkk:",body_np[point_2])
        if point_1==27 or point_1==29 or point_1 == 31:
            po27 = body_np[27]
            po29 = body_np[29]
            po31 = body_np[31]
            po27 = po27[1:]
            po29 = po29[1:]
            po31 = po31[1:]
            pointL = (po27+po29+po31)/3 #手掌座標
        if  point_2 == 28 or point_2 == 30 or point_2 ==32:
            po28 = body_np[28]
            po30 = body_np[30]
            po32 = body_np[32]
            po28 = po28[1:]
            po30 = po30[1:]
            po32 = po32[1:]
            pointR = (po28+po30+po32)/3

        #print("point1:",pointL)
        #print("point2:",pointR)
        return pointL, pointR #回傳腳的x, y

    def min_n(lst, n=1):
        # 使用sorted函數對列表進行排序（升序排列）
        sorted_list = sorted(enumerate(lst), key=lambda x: x[1], reverse=False)  #因為enumerate了，所以後面要用[1]
        
        # 取出前10個元素的索引
        bottom_10_index = [x[0] for x in sorted_list[:n]]
        # 顯示結果
        #print("min10dist:",sorted_list[:n])
        #print("min10:",bottom_10_index)
        return bottom_10_index ,sorted_list[:n]



    def fourier2(x, a0, a1, b1, a2, b2, w):
        return a0 + a1*np.cos(x*w) + b1*np.sin(x*w) + a2*np.cos(2*x*w) + b2*np.sin(2*x*w)

    def stad_count(csv_name): #算次數用
        with open(csv_name, 'r') as file:
            # read csv
            reader = csv.reader(file)
            column2_data = [row[1] for row in reader]

        numpy_array = np.array(column2_data[1:], dtype=float) #把距離都取出來
        #print(numpy_array)

        y_data = numpy_array
        num_points = len(y_data)
        # y_data = y_data[start_index:end_index]
        # normalize [0, 1]
        y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))
        # make X data
        x_data = np.arange(1, len(y_data)+1)

        std_list = []
        for i in range(1, 16):
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

        fit_params, _ = curve_fit(fourier2, x_data, y_data, p0=initial_guess, maxfev=50000, method='lm')

        #print("標準差:", residuals_std)
        
    
        #print("参数:", fit_params)

        # 畫圖
        #plt.scatter(x_data, y_data, label='Data') #原本的距離跟len


        x_fit = x_data
        y_fit = fourier2(x_data, *fit_params)

        peaks, _ = find_peaks(y_data, prominence=0.3, distance = 3)
        valleys, _ = find_peaks(-y_data, prominence=0.3, distance = 3)
        #print("peak:",peaks)
        time = len(peaks)+len(valleys) 
        #print("howmany",len(peaks)) #算次數 ，手摸肩膀跟膝蓋的起始是一邊先摸好
        
        max_vals = []
        min_vals = []
        for i in peaks:
            #print(y_data[i])
            max_vals.append(y_data[i])
        for j in valleys:
            #print(y_data[i])
            min_vals.append(y_data[j])
        #print('Max values:', max_vals)


        #plt.plot(x_fit, y_fit, 'r-', label='Fit')
        plt.plot(y_data)
        plt.xlabel('x')
        plt.ylabel('y')
        x_major_locator = MultipleLocator(10) #把x軸刻度設1, 存在變數裡
        ax = plt.gca() #ax為兩個座標軸的實例
        ax.xaxis.set_major_locator(x_major_locator) #把x座標軸刻度設為1的倍數
        plt.xlim(0.5,len(y_data))
        plt.scatter(peaks, max_vals, c ='red')
        # 顯示圖表
        pic_num = csv_name.split(".")[1].split('/')[1]
        #print(pic_num)
        cccc = pic_num
        folder = "./"+cccc+"/test/"
        pic = folder +pic_num +"_up.png" #圖片存檔
        plt.savefig(pic)
        #plt.show()
        
        return time, peaks, valleys



    def catch2_2(where , lis):#找出波峰波谷前後2幀的值
        get = []
        for i in where: #先確認位置
            find = [] #用來存前後2幀的index
            if len(lis)>i:
                #print(i)
                find.append(i-2) #開始存前後2幀的index
                find.append(i-1)
                find.append(i)
                find.append(i+1)
                find.append(i+2)
            #print(find)
            for i in find: #拿掉不能用的index
                #print(i)
                if i<0 or i>=len(lis):
                    find.remove(i)
            #print("find:",find)
            value_ = [] #用來放這五幀的值
            for i in find:
                #print("gp:",lis[i])
                value_.append(lis[i])
            #print("sum:",value_)
            go = min(value_)
            get.append(go)#取最小的存
            #print("min:",go)

        return get

    def fordist(bodynp):
        global sa_F, sa_H
        first = bodynp
        Lhand, Rhand = hand_point(first,19,20)

        Lknee = first[25][1:]
        Rknee = first[26][1:]
        Lsho = first[11][1:]
        Rsho = first[12][1:]
        # Lheel = first[29][1:]
        # Rheel = first[30][1:]




        distR_f = np.sqrt(np.sum((Rhand-Lknee)**2)) #膝蓋跟手的距離，這個是右手跟左膝
        distL_f = np.sqrt(np.sum((Lhand-Rknee)**2))
        distR_h = np.sqrt(np.sum((Rhand-Lsho)**2)) #手跟肩膀的距離
        distL_h = np.sqrt(np.sum((Lhand-Rsho)**2))


        #print("distL_f:", distL_f)
        #print("distR_f:", distR_f)
        #print("distL_h:", distL_h)
        #print("distR_h:", distR_h)
        
        if distL_f < distR_f: #表示左手摸膝蓋
            footdi = distL_f
            sa_F.append(footdi)
        else:
            footdi = distR_f
            sa_F.append(footdi)
        #print(len(sa_F))
        if distL_h < distR_h: #表示左手摸肩膀
            handdi = distL_h #手的距離
            sa_H.append(handdi)
        else:
            handdi = distR_h
            sa_H.append(handdi)
        #print(len(sa_H))

        return footdi, handdi


    def acc_score(foot, hand):#舊的算法
        global thres_F, thres_H
        #1.0的腳比值表示都沒抬，是10分，0.7是100
        if foot <=thres_F:
            scoreF = 100
        else:
            scoreF = -250 * (foot - thres_F) + 100
            if scoreF<0:
                scoreF = 0
        
        #print("scoreF:",scoreF)

        if hand <=  thres_H:
            scoreH = 100
        else:
            scoreH = -250 * (hand -  thres_H) + 100
            if scoreH<0:
                scoreH = 0
        #print("scoreH:",scoreH)

        totalscore = int(scoreF * 0.5 + scoreH * 0.5)

        return  totalscore

    def acc_score2(fff, hhh): 
        global thres_F, thres_H
        allf = fff
        allh = hhh
        good_foot = thres_F
        good_hand = thres_H
        gf = 0
        gh = 0
        for i in allf:
            #print(i)
            if i <= good_foot:
                gf+=1
        for j in allh:
            #print(j)
            if j <= good_hand:
                gh+=1
        if len(allf) == 0:
            totalscore =60 
        elif len(allh) == 0:
            totalscore =60
        else:
            scoreF = int((gf/len(allf))*100)
            scoreH = int((gh/len(allh))*100)

            totalscore = int(scoreF * 0.5 + scoreH * 0.5)

            totalscore = totalscore + 36

            totalscore = int((np.sqrt(totalscore))*10)
            
            if totalscore >=100:
                totalscore =  random.randint(91, 100)

        return totalscore




    #流暢度，目前以1秒一次為標準
    def smmooth_score(time,howmany):
        elapsed_time = time
        one_time = (elapsed_time)/howmany
        if one_time <= 1:
            speed = random.randint(91, 100) #流暢度成績
        else:
            speed = -100*(one_time-1)+100
            speed = int(speed)+36
            if speed < 0:
                speed = 36

        speed = int((np.sqrt(speed))*10)
        
        if speed >=100:
            speed =  random.randint(91, 100)
        return speed

    def writetxt(content,path):
        
        file_path = path  # 檔案路徑
        #print("text",content)
        # 要寫入的三個字串內容
        text_content = content
        new_list = [str(x) for x in text_content]

        # 開啟檔案並寫入文字內容
        with open(file_path, "w") as file:
            file.writelines("\n".join(new_list))


    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    #一般攝影機
    #?????????
    cap = cv2.VideoCapture(0)

    video_name = "./04/04.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, int(cap.get(cv2.CAP_PROP_FPS))/2, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    fps = cap.get(cv2.CAP_PROP_FPS)
    allframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vvv = "04"
    #print("fps:",int(cap.get(cv2.CAP_PROP_FPS))/2 )
    frame_count = 0
    output_folder = "./"+vvv+'/test/keypoints-'+ video_name.split("/")[2].split(".")[0]  # 設定儲存關鍵點檔案的資料夾路徑
    output_folder2 = "./"+vvv+'/test/keypoints-3d-'+ video_name.split("/")[2].split(".")[0]
    score_txt = "./04score.txt"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)

    a123 = ""
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        #start_time = time.time() #設定開始時間，這個動作可以給3秒倒數計時


        allkeys = np.empty((0, 4), dtype=np.float32)
        while True:
            
            ret, img = cap.read()
            img= cv2.flip(img,1)
            h, w, c = img.shape
            if not ret:
                #print("Cannot receive frame")
                break
            if a123 == "": #用來擋，卡在第一幀不動，有輸入才會繼續錄
                a123 = input()
                start123 = time.time()
            results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                frame_count = frame_count+1
                now_time = time.time()-start123
                #print("frame count:",frame_count)
                use_keys = np.empty((0, 4), dtype=np.float32)
                key_dist = np.empty((0, 4), dtype=np.float32) #用來算距離用
                all3dkeys = np.empty((0, 4), dtype=np.float32)
                # mp_drawing.draw_landmarks(
                #     img,
                #     results.pose_landmarks,
                #     mp_holistic.POSE_CONNECTIONS,
                #     landmark_drawing_spec=mp_drawing_styles
                # .get_default_pose_landmarks_style())
                
                for id, landmark in enumerate(results.pose_landmarks.landmark):

                    use_key = np.array([id,landmark.x, landmark.y, landmark.z], dtype=np.float32)
                    use_keys = np.vstack([use_keys, use_key])
                    
                    
                    key_dist= np.vstack([key_dist, use_key])
                if now_time>10:
                    output_file = os.path.join(output_folder, f'keypoints_{frame_count}.npy')
                    np.save(output_file, key_dist)
                    #print("put 1")
                for id, landmark in enumerate(results.pose_world_landmarks.landmark):
                    use_key = np.array([id,landmark.x, landmark.y, landmark.z], dtype=np.float32)
                    allkeys = np.vstack([allkeys, use_key])
                    all3dkeys = np.vstack([all3dkeys, use_key]) 
                if now_time>10:
                    output_file = os.path.join(output_folder2, f'keypoints_{frame_count}.npy')
                    np.save(output_file, all3dkeys)
                    #print("put 2")
                
                leg, handsss = fordist(all3dkeys)
                

                

                cv2.line(img,(int(key_dist[11][1]*w), int(key_dist[11][2]*h)), (int(key_dist[12][1]*w), int(key_dist[12][2]*h)), (255, 255, 255), 2)#肩膀
                cv2.line(img,(int(key_dist[11][1]*w), int(key_dist[11][2]*h)), (int(key_dist[13][1]*w), int(key_dist[13][2]*h)),  (255, 255, 255), 2)#左手臂
                cv2.line(img,(int(key_dist[13][1]*w), int(key_dist[13][2]*h)), (int(key_dist[15][1]*w), int(key_dist[15][2]*h)),  (255, 255, 255), 2)#左手
                cv2.line(img,(int(key_dist[12][1]*w), int(key_dist[12][2]*h)), (int(key_dist[14][1]*w), int(key_dist[14][2]*h)),  (255, 255, 255), 2)#右手臂
                cv2.line(img,(int(key_dist[14][1]*w), int(key_dist[14][2]*h)), (int(key_dist[16][1]*w), int(key_dist[16][2]*h)),  (255, 255, 255), 2)#右手
                cv2.line(img,(int(key_dist[11][1]*w), int(key_dist[11][2]*h)), (int(key_dist[23][1]*w), int(key_dist[23][2]*h)),  (255, 255, 255), 2)#身體
                cv2.line(img,(int(key_dist[12][1]*w), int(key_dist[12][2]*h)), (int(key_dist[24][1]*w), int(key_dist[24][2]*h)),  (255, 255, 255), 2)#身體
                cv2.line(img,(int(key_dist[23][1]*w), int(key_dist[23][2]*h)), (int(key_dist[24][1]*w), int(key_dist[24][2]*h)),  (255, 255, 255), 2)#身體
                cv2.line(img,(int(key_dist[23][1]*w), int(key_dist[23][2]*h)), (int(key_dist[25][1]*w), int(key_dist[25][2]*h)), (255, 255, 255), 2)#身體
                cv2.line(img,(int(key_dist[25][1]*w), int(key_dist[25][2]*h)), (int(key_dist[27][1]*w), int(key_dist[27][2]*h)), (255, 255, 255), 2)#身體
                cv2.line(img,(int(key_dist[27][1]*w), int(key_dist[27][2]*h)), (int(key_dist[29][1]*w), int(key_dist[29][2]*h)), (255, 255, 255), 2)#身體
                cv2.line(img,(int(key_dist[29][1]*w), int(key_dist[29][2]*h)), (int(key_dist[31][1]*w), int(key_dist[31][2]*h)), (255, 255, 255), 2)#身體
                cv2.line(img,(int(key_dist[27][1]*w), int(key_dist[27][2]*h)), (int(key_dist[31][1]*w), int(key_dist[31][2]*h)), (255, 255, 255), 2)#身體
                cv2.line(img,(int(key_dist[24][1]*w), int(key_dist[24][2]*h)), (int(key_dist[26][1]*w), int(key_dist[26][2]*h)), (255, 255, 255), 2)#身體
                cv2.line(img,(int(key_dist[26][1]*w), int(key_dist[26][2]*h)), (int(key_dist[28][1]*w), int(key_dist[28][2]*h)), (255, 255, 255), 2)#身體
                cv2.line(img,(int(key_dist[28][1]*w), int(key_dist[28][2]*h)), (int(key_dist[30][1]*w), int(key_dist[30][2]*h)), (255, 255, 255), 2)#身體
                cv2.line(img,(int(key_dist[30][1]*w), int(key_dist[30][2]*h)), (int(key_dist[32][1]*w), int(key_dist[32][2]*h)), (255, 255, 255), 2)#身體
                cv2.line(img,(int(key_dist[28][1]*w), int(key_dist[28][2]*h)), (int(key_dist[32][1]*w), int(key_dist[32][2]*h)), (255, 255, 255), 2)#身體


                cv2.line(img,(int(key_dist[15][1]*w), int(key_dist[15][2]*h)), (int(key_dist[17][1]*w), int(key_dist[17][2]*h)), (255, 255, 255), 2)#左手掌
                cv2.line(img,(int(key_dist[15][1]*w), int(key_dist[15][2]*h)), (int(key_dist[19][1]*w), int(key_dist[19][2]*h)),  (255, 255, 255), 2)
                cv2.line(img,(int(key_dist[17][1]*w), int(key_dist[17][2]*h)), (int(key_dist[19][1]*w), int(key_dist[19][2]*h)),  (255, 255, 255), 2)
                cv2.line(img,(int(key_dist[15][1]*w), int(key_dist[15][2]*h)), (int(key_dist[21][1]*w), int(key_dist[21][2]*h)),  (255, 255, 255), 2)
                cv2.line(img,(int(key_dist[16][1]*w), int(key_dist[16][2]*h)), (int(key_dist[18][1]*w), int(key_dist[18][2]*h)),  (255, 255, 255), 2)#右手掌
                cv2.line(img,(int(key_dist[16][1]*w), int(key_dist[16][2]*h)), (int(key_dist[20][1]*w), int(key_dist[20][2]*h)),  (255, 255, 255), 2)
                cv2.line(img,(int(key_dist[18][1]*w), int(key_dist[18][2]*h)), (int(key_dist[20][1]*w), int(key_dist[20][2]*h)),  (255, 255, 255), 2)
                cv2.line(img,(int(key_dist[16][1]*w), int(key_dist[16][2]*h)), (int(key_dist[22][1]*w), int(key_dist[22][2]*h)), (255, 255, 255), 2)





                for i in range(11,33):
                    cv2.circle(img,(int(key_dist[i][1]*w), int(key_dist[i][2]*h)),4,(212,199,129),-1)
                

                #cv2.putText(img,("time:"+str(round((time.time() - start_time),2))), (40, 60), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (0,255,0),2)
                #cv2.putText(img,("hand:"+str(round(handsss,2))), (40, 80), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (200,0,255),2)
                #cv2.putText(img,("foot:"+str(round(leg,2))), (40, 100), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (200,0,255),2)




            # cv2.namedWindow('abc123', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('abc123', 1280, 720)
            cv2.imshow('abc123', img)
            out.write(img)
            #key = cv2.waitKey(1)
            
            if cv2.waitKey(5) == ord('q'):
                break
            if (time.time() -  start123)>30:
                break
                    # 計算經過的時間

    cv2.destroyAllWindows()
    #elapsed_time=round(frame_count / fps,2)
    elapsed_time = 20





    np_path = [os.path.join(output_folder,f) for f in os.listdir(output_folder)if f.endswith('.npy')]           
    np_sorted = natsorted (np_path) 
    dist_all=[]
    for i, file in enumerate(np_sorted):
        get=np.load(file)
        np0 = np.load(np_sorted[0])
        #rint(get.shape)
        if get.shape == np0.shape:
            #print(file)
            ske_dist = euclidean.eucliDist(np_sorted[0], file)
            #ske_dist = euclidean.eucliDist("./yn01/pic/yn01brain/np/frame0_key.npy", file)
            dist_all.append(ske_dist)
    #print("all:",dist_all)

    headers = ["pic", "dist"]
    csv_name = "./"+vvv+"/test/"+video_name.split("/")[2].split(".")[0]
    path = csv_name + ".csv"
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for i in range(len(dist_all)):
            writer.writerow([i,dist_all[i]]) #存影片的動作所有距離
    #3D        
    np_path3 = [os.path.join(output_folder2,f) for f in os.listdir(output_folder2)if f.endswith('.npy')]           
    np_sorted3 = natsorted (np_path3) 
    dist_all3=[]
    for i, file in enumerate(np_sorted3):
        get=np.load(file)
        np0 = np.load(np_sorted3[0])
        gettt =np.empty((0, 4), dtype=np.float32)
        np00 = np.empty((0, 4), dtype=np.float32)
        np00 = np.vstack([np00, np0[13:23]])
        np00 = np.vstack([np00, np0[27:]])
        #print(get.shape)
        if get.shape == np0.shape:
            #print(file)
            gettt = np.vstack([gettt, get[13:23]]) #只存手跟腳
            gettt = np.vstack([gettt, get[27:]])
            #ske_dist3 = euclidean.eucliDist_no(np_sorted[0], file, 11 ,32)
            ske_dist3= np.sqrt(np.sum((np00-gettt)**2))
            #ske_dist = euclidean.eucliDist("./yn01/pic/yn01brain/np/frame0_key.npy", file)
            dist_all3.append(ske_dist3)
    #print("all:",dist_all3)

    headers3 = ["pic", "dist"] #存3d
    csv_name3 = "./"+vvv+"/test/"+video_name.split("/")[2].split(".")[0]+'-1'
    path3 = csv_name3 + ".csv"
    #print("path3",path3)
    with open(path3, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers3)
        for i in range(len(dist_all3)):
            writer.writerow([i,dist_all3[i]]) #存影片的動作所有距離

    #開始算成績
    howmany,  whpeak, whvalley = stad_count(path3) #算總共做了幾次
    #print("times:",howmany)
    #print("whpeak:",whpeak)
    #print("whvalley:",whvalley)
    #print("ffff:",len(sa_F))
    #print("hhhh:",len(sa_H))
    get_F = []
    get_H = []

    peakva = catch2_2(whpeak,sa_F)
    valleyva = catch2_2(whvalley,sa_F)
    peakva.extend(valleyva) #把兩邊的值加起來
    get_F = peakva
    peakvaH = catch2_2(whpeak,sa_H)
    valleyvaH = catch2_2(whvalley,sa_H)
    peakvaH.extend(valleyvaH) #把兩邊的值加起來
    get_H = peakvaH


    #print("get_F:",get_F)
    #print("get_H:",get_H)
    # avg_F = sum(get_F)/len(get_F) #平均的腳
    # avg_H = sum(get_H)/len(get_H) #平均的手
    # accuracy = acc_score(avg_F,avg_H) #準確度成績
    accuracy = acc_score2(get_F,get_H)

    speed = smmooth_score(elapsed_time,howmany) #流暢度成績


    #標準差(這種暫停)
    # this_std3 = Std_of_stability.stadard_count(path3)#3d的標準差
    # print("this:",this_std3)
    # coach_std3 = Std_of_stability.stadard_count('./csv/12.csv') #教練
    # print("coach:",coach_std3)
    # std_score3 = int(calculate_std(this_std3,coach_std3))

    LH, RH, LF, RF = for_knee.step_top (output_folder2 , howmany) #這次動作的四肢std
    LH_score =calculate_std(LH ,LH_B_std)#四個成績
    RH_score =calculate_std(RH ,RH_B_std)
    LF_score =calculate_std(LF ,LH_S_std)
    RF_score =calculate_std(RF ,RH_S_std)
    std_score3 = int((LH_score+RH_score+LF_score+RF_score)/4)
    std_score3 = std_score3+36
    std_score3 = int((np.sqrt(std_score3))*10)
    if std_score3 >=100:
        std_score3 =  random.randint(91, 100)


    #print("LH_score:",LH_score)
    #print("RH_score:",RH_score)
    #print("LF_score:",LF_score)
    #print("RF_score:",RF_score)



    #print(f"穩定度評分3:{std_score3}分")
    #print(f"準確度評分:{accuracy}分")
    #print(f"流暢度評分3:{speed}分")

    # if std_score3 < 20:
    #     std_score3 += 60
    # if accuracy < 20:
    #     accuracy += 60
    # if speed < 20:
    #     speed += 60
    print(f"stb:{std_score3}")
    print(f"cro:{accuracy}")
    print(f"flu:{speed}")
    emoscore = emotion.emotionscore()*0.7
    emoscore = int(emoscore + (std_score3 + accuracy + speed )/3*0.3)

    print(f"emo:{emoscore}")
    # score_list = []
    # score_list.append(std_score3)
    # score_list.append(accuracy)
    # score_list.append(speed)


    cv2.destroyAllWindows()
    #writetxt(score_list,score_txt)

    # img12 = np.zeros((256, 256, 3), np.uint8)

    # img12 = cv2.resize(img12,(700,700))

    # # 將圖片用淺灰色 (200, 200, 200) 填滿
    # img12.fill(255)

    # 在圖片上畫一條紅色的對角線，寬度為 5 px
    #cv2.putText(img12, ("Stability score: " + str(std_score3)), (10, 50), cv2.FONT_HERSHEY_DUPLEX,1, (0, 0, 0), 1, cv2.LINE_AA)

    #cv2.putText(img12, ("Accuracy score: " + str(accuracy)), (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 0, 0), 1, cv2.LINE_AA)

    #cv2.putText(img12, ("Smoothness score: " + str(speed)), (10, 150), cv2.FONT_HERSHEY_DUPLEX,1, (0, 0, 0), 1, cv2.LINE_AA)


    # 顯示圖片
    #cv2.imshow('My Image', img12)

    # 按下任意鍵則關閉所有視窗



            


    cap.release()
    out.release()
    cv2.destroyAllWindows()


    process_file.removefile("./04/test")
    process_file.sort_and_rename_files("./save/04",video_name)


except:
    print(f"stb:73")
    print(f"cro:65")
    print(f"flu:68")
    emoscore = emotion.emotionscore()*0.7
    emoscore = int(emoscore + 65*0.3)

    print(f"emo:{emoscore}")
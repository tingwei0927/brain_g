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
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import csv
import os
import euclidean
import Std_of_stability
from matplotlib.pyplot import MultipleLocator 
import time
import shutil
import for_out
import process_file
import for_knee 
import for_step

cap = cv2.VideoCapture(0)
nnn = input()
if nnn == "1":
    #手掌拍肩     
    try:
        sa_H = [] #存手的距離成績
        sa_angle = []
        thres_H = 0.55
        angle_thres = 30
        #目前暫時用角度來看，角度只用x,y，看到peak的那幾幀的最小和45度的差距
        #下面四個是教練的四肢標準差
        LH_S_std=0.0501
        RH_S_std=0.058
        LH_B_std=0.0037
        RH_B_std=0.0075




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
                pointL = (po17+po19)/2 #手掌座標
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
            #print("howmany",len(peaks)) #算次數 ，踏步的開始必須是站好，不然會出問題
            
            max_vals = []
            min_vals = []
            for i in peaks:
                #print(y_data[i])
                max_vals.append(y_data[i])
            for j in valleys:
                #print(y_data[i])
                min_vals.append(y_data[j])
        # print('Max values:', max_vals)


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


        def ang(x1,y1,x2,y2,x3,y3):
            # 計算三個邊的平方
            AB_squared = (x2 - x1)**2 + (y2 - y1)**2 
            length_A = math.sqrt(AB_squared)
            BC_squared = (x3 - x2)**2 + (y3 - y2)**2 
            length_B = math.sqrt(BC_squared)
            AC_squared = (x3 - x1)**2 + (y3 - y1)**2 
            length_C = math.sqrt(AC_squared)
            # 計算夾角的餘弦值
            
            
            
            # if (length_A + length_B) <= length_C or (length_A + length_C) <= length_B or (length_B + length_C) <= length_A:
            #     theta_deg  = "< third line"
            #     return theta_deg
            # else:
            cos_theta = (AB_squared + BC_squared - AC_squared) / (2 * length_A * length_B)
                # 檢查 cos_theta 的值是否在有效範圍內
                # if cos_theta < -1.0 or cos_theta > 1.0:
                #     theta_deg = "<-1 or >1"
                #     return theta_deg

                # 計算夾角的弧度
            theta_rad = math.acos(cos_theta)

                # 轉換為度數
            theta_deg = round(math.degrees(theta_rad),2)

            return theta_deg


        def fordist(bodynp): #算手的距離跟另一邊的手打開的角度
            global  sa_H, sa_angle
            first = bodynp

            Lsho = first[11][1:]
            Rsho = first[12][1:]
            Lhip = first[23][1:]
            Rhip = first[24][1:]
            Rel = first[14][1:]
            Lel = first[13][1:]
            lhand,rhand = hand_point(first, 17, 18) #左右手坐標
            R_angle = ang(Rel[0],Rel[1],Rsho[0],Rsho[1],Rhip[0],Rhip[1]) #右邊正面看角度
            L_angle = ang(Lel[0],Lel[1],Lsho[0],Lsho[1],Lhip[0],Lhip[1])
            
            
            distL_h = np.sqrt(np.sum((lhand-Rsho)**2)) #手跟肩膀的距離
            distR_h = np.sqrt(np.sum((rhand-Lsho)**2))


            #print("distL_h:", distL_h)
            #print("distR_h:", distR_h)
            

            if distL_h < distR_h: # 表示現在是左手摸右肩
                handdi = distL_h #手的距離
                angle = R_angle
                sa_H.append(handdi)
                sa_angle.append(angle)
            else:
                handdi = distR_h
                angle = L_angle
                sa_H.append(handdi)
                sa_angle.append(angle)
                
            #print(len(sa_H))
            #print(len(sa_angle))

            return handdi, angle






        def catch2_2(where , lis):#找出波峰波谷前後2幀的值
            get = []
            for i in where: #先確認位置
                find = [] #用來存前後2幀的index
                if len(lis)>i:
            #     print(i)
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


        def acc_score(angle, hand):
            global thres_H, angle_thres
            #1.0的腳比值表示都沒抬，是10分，0.7是100
            if angle <= angle_thres:
                scoreF = 100
            else:
                scoreF = -250 * (angle -angle_thres) + 100
                if scoreF<0:
                    scoreF = 0
            
            #print("scoreF:",scoreF)

            if hand <= thres_H:
                scoreH = 100
            else:
                scoreH = -250 * (hand - thres_H) + 100
                if scoreH<0:
                    scoreH = 0
            #print("scoreH:",scoreH)

            totalscore = int(scoreF * 0.5 + scoreH * 0.5)

            return  totalscore

        def acc_score2(aaa, hhh): #先放著，不一定用，這個就是看有幾次做到標準
            global thres_H, angle_thres
            alla = aaa
            allh = hhh
            good_angle = angle_thres
            good_hand = thres_H
            ga = 0
            gh = 0
            for i in alla:
                #print(i)
                if i >= good_angle:
                    ga+=1
            for j in allh:
                #print(j)
                if j <= good_hand:
                    gh+=1

            scoreA = int((ga/len(alla))*100)
            scoreH = int((gh/len(allh))*100)

            totalscore = int(scoreA * 0.5 + scoreH * 0.5)

            totalscore = totalscore + 36

            totalscore = int((np.sqrt(totalscore))*10)
            
            if totalscore >=100:
                totalscore = 100

            return totalscore




        #流暢度，目前以1秒一次為標準
        def smmooth_score(time,howmany):
            elapsed_time = time
            one_time = (elapsed_time-5)/howmany
            if one_time <= 1: #流暢度暫定秒數
                speed =100 #流暢度成績
            else:
                speed = -100*(one_time-1)+100
                speed = int(speed)+36
                if speed < 0:
                    speed = 36

            speed = int((np.sqrt(speed))*10)
            
            if speed >=100:
                speed = 100
                
            return speed

            





        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic




        #一般攝影機
        #cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # cap.set(cv2.CAP_PROP_FPS, 30)

        video_name = "./01/01.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_name, fourcc, int(cap.get(cv2.CAP_PROP_FPS))/2, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        fps = cap.get(cv2.CAP_PROP_FPS)
        allframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        vvv = "01"
        frame_count = 0
        tall=[]
        output_folder = "./"+vvv+'/test/keypoints-'+ video_name.split("/")[2].split(".")[0]  # 設定儲存關鍵點檔案的資料夾路徑
        output_folder2 = "./"+vvv+'/test/keypoints-3d-'+ video_name.split("/")[2].split(".")[0]
        score_txt = "./01score.txt"
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(output_folder2, exist_ok=True)

        a123 = ""

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
                
            allkeys = np.empty((0, 4), dtype=np.float32)
            
            
            while True:
                
                ret, img = cap.read()
                img= cv2.flip(img,1)
                h, w, c = img.shape
                if not ret:
                    #print("Cannot receive frame")
                    break
                if a123 == "": #用來擋，卡在第一幀不動，有輸入才會繼續錄
                    a123 = "123"
                    start123 = time.time()
                
                results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                if results.pose_landmarks:
                    frame_count = frame_count+1
                    #print("frame count:",frame_count)
                    use_keys = np.empty((0, 4), dtype=np.float32)
                    key_dist = np.empty((0, 4), dtype=np.float32) #用來算距離用
                    all3dkeys = np.empty((0, 4), dtype=np.float32)
                    mp_drawing.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
                    
                    for id, landmark in enumerate(results.pose_landmarks.landmark):

                        use_key = np.array([id,landmark.x, landmark.y, landmark.z], dtype=np.float32)
                        use_keys = np.vstack([use_keys, use_key])
                        key_dist= np.vstack([key_dist, use_key])
                        output_file = os.path.join(output_folder, f'keypoints_{frame_count}.npy')
                        np.save(output_file, key_dist)

                            #print("put 1")
                    for id, landmark in enumerate(results.pose_world_landmarks.landmark):
                        use_key = np.array([id,landmark.x, landmark.y, landmark.z], dtype=np.float32)
                        allkeys = np.vstack([allkeys, use_key])
                        all3dkeys = np.vstack([all3dkeys, use_key]) 

                        output_file = os.path.join(output_folder2, f'keypoints_{frame_count}.npy')
                        np.save(output_file, all3dkeys)
                        #print("put 2")
                    
                    if frame_count<31:
                        LSHdis = np.sqrt(np.sum((all3dkeys[23][1:]-all3dkeys[31][1:])**2))
                        tall.append(LSHdis)
                    tall__ = sum(tall)/len(tall)
                    #print("tall:",tall__)

                    handsss, angle = fordist(all3dkeys)


                    
                    #cv2.putText(img,("hand:"+str(round(handsss,2))), (40, 80), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (200,0,255),2)
                    #cv2.putText(img,("angle:"+str(round(angle,2))), (40, 100), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (200,0,255),2)
                    


                    

                                


                
                cv2.namedWindow('abc123', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('abc123', 1280, 720)
                cv2.imshow('abc123', img)
                out.write(img)
                if cv2.waitKey(5) == ord('q'):
                    break
                if (time.time() -  start123 )>30:
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        elapsed_time=round(frame_count / fps,2)
            
        np_path = [os.path.join(output_folder,f) for f in os.listdir(output_folder)if f.endswith('.npy')]           
        np_sorted = natsorted (np_path) 
        dist_all=[]
        for i, file in enumerate(np_sorted):
            get=np.load(file)
            np0 = np.load(np_sorted[0])
            #print(get.shape)
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
        # np_path3 = [os.path.join(output_folder2,f) for f in os.listdir(output_folder2)if f.endswith('.npy')]           
        # np_sorted3 = natsorted (np_path3) 
        # dist_all3=[]
        # for i, file in enumerate(np_sorted3):
        #     get=np.load(file)
        #     np0 = np.load(np_sorted3[0])
        #     #print(get.shape)
        #     if get.shape == np0.shape:
        #         #print(file)
        #         ske_dist = euclidean.eucliDist(np_sorted3[0], file)
        #         #ske_dist = euclidean.eucliDist("./yn01/pic/yn01brain/np/frame0_key.npy", file)
        #         dist_all3.append(ske_dist)
        # #print("all:",dist_all3)

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


        howmany, whpeak, whvalley = stad_count(path3) #算總共做了幾次
        #print("times:",howmany)
        #print("whpeak:",whpeak)
        #print("whvalley:",whvalley)
        #print("aaaa:",len(sa_angle))
        #print("hhhh:",len(sa_H))
        get_A = []
        get_H = []

        peakva = catch2_2(whpeak,sa_angle)
        valleyva = catch2_2(whvalley,sa_angle)
        peakva.extend(valleyva) #把兩邊的值加起來
        get_A = peakva
        peakvaH = catch2_2(whpeak,sa_H)
        valleyvaH = catch2_2(whvalley,sa_H)
        peakvaH.extend(valleyvaH) #把兩邊的值加起來
        get_H = peakvaH
        #print("get_A:",get_A)
        #print("get_H:",get_H)
        # avg_F = sum(get_F)/len(get_F) #平均的腳
        # avg_H = sum(get_H)/len(get_H) #平均的手
        # accuracy = acc_score(avg_F,avg_H) #準確度成績
        accuracy = acc_score2(get_A,get_H)

        speed = smmooth_score(elapsed_time,howmany) #流暢度成績


        LH, RH, LF, RF = for_out.step_top (output_folder2,int(howmany/2)) #這次動作的四肢std
        LH_score =calculate_std(LH,LH_S_std)#四個成績
        RH_score =calculate_std(RH,RH_S_std)
        LF_score =calculate_std(LF,LH_B_std)
        RF_score =calculate_std(RF,RH_B_std)
        std_score3 = int((LH_score+RH_score+LF_score+RF_score)/4)
        std_score3 = std_score3+36
        std_score3 = int((np.sqrt(std_score3))*10)
        if std_score3 >=100:
            std_score3 = 100

        print(f"stb:{std_score3}")
        print(f"cro:{accuracy}")
        print(f"flu:{speed}")
                    


        process_file.removefile("./01/test")
        process_file.sort_and_rename_files("./save/01",video_name)

    except:
        print(f"stb:65")
        print(f"cro:65")
        print(f"flu:65")
                
elif nnn == "4":
        #交互拍肩
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

            distR_f = np.sqrt(np.sum((Rhand-Lknee)**2)) #膝蓋跟手的距離，這個是右手跟左膝
            distL_f = np.sqrt(np.sum((Lhand-Rknee)**2))
            distR_h = np.sqrt(np.sum((Rhand-Lsho)**2)) #手跟肩膀的距離
            distL_h = np.sqrt(np.sum((Lhand-Rsho)**2))
            
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
                    totalscore = 100

            return totalscore




        #流暢度，目前以1秒一次為標準
        def smmooth_score(time,howmany):
            elapsed_time = time
            one_time = (elapsed_time-5)/howmany
            if one_time <= 1:
                speed =100 #流暢度成績
            else:
                speed = -100*(one_time-1)+100
                speed = int(speed)+36
                if speed < 0:
                    speed = 36

            speed = int((np.sqrt(speed))*10)
            
            if speed >=100:
                speed = 100
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
        #cap = cv2.VideoCapture(0)

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
                    a123 = "123"
                    start123 = time.time()
                results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    frame_count = frame_count+1
                    #print("frame count:",frame_count)
                    use_keys = np.empty((0, 4), dtype=np.float32)
                    key_dist = np.empty((0, 4), dtype=np.float32) #用來算距離用
                    all3dkeys = np.empty((0, 4), dtype=np.float32)
                    mp_drawing.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
                    
                    for id, landmark in enumerate(results.pose_landmarks.landmark):

                        use_key = np.array([id,landmark.x, landmark.y, landmark.z], dtype=np.float32)
                        use_keys = np.vstack([use_keys, use_key])
                        
                        
                        key_dist= np.vstack([key_dist, use_key])
                        output_file = os.path.join(output_folder, f'keypoints_{frame_count}.npy')
                        np.save(output_file, key_dist)
                        #print("put 1")
                    for id, landmark in enumerate(results.pose_world_landmarks.landmark):
                        use_key = np.array([id,landmark.x, landmark.y, landmark.z], dtype=np.float32)
                        allkeys = np.vstack([allkeys, use_key])
                        all3dkeys = np.vstack([all3dkeys, use_key]) 
                        
                        output_file = os.path.join(output_folder2, f'keypoints_{frame_count}.npy')
                        np.save(output_file, all3dkeys)
                        #print("put 2")
                    
                    leg, handsss = fordist(all3dkeys)

                cv2.namedWindow('abc123', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('abc123', 1280, 720)
                cv2.imshow('abc123', img)
                out.write(img)
                #key = cv2.waitKey(1)
                
                if cv2.waitKey(5) == ord('q'):
                    break
                if (time.time() -  start123)>30:
                    break
                        # 計算經過的時間

        cv2.destroyAllWindows()
        elapsed_time=round(frame_count / fps,2)
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
        accuracy = acc_score2(get_F,get_H)

        speed = smmooth_score(elapsed_time,howmany) #流暢度成績

        LH, RH, LF, RF = for_knee.step_top (output_folder2 , howmany) #這次動作的四肢std
        LH_score =calculate_std(LH ,LH_B_std)#四個成績
        RH_score =calculate_std(RH ,RH_B_std)
        LF_score =calculate_std(LF ,LH_S_std)
        RF_score =calculate_std(RF ,RH_S_std)
        std_score3 = int((LH_score+RH_score+LF_score+RF_score)/4)
        std_score3 = std_score3+36
        std_score3 = int((np.sqrt(std_score3))*10)
        if std_score3 >=100:
            std_score3 = 100

        print(f"stb:{std_score3}")
        print(f"cro:{accuracy}")
        print(f"flu:{speed}")


        cv2.destroyAllWindows()


        cap.release()
        out.release()
        cv2.destroyAllWindows()


        process_file.removefile("./04/test")
        process_file.sort_and_rename_files("./save/04",video_name)


    except:
        print(f"stb:65")
        print(f"cro:65")
        print(f"flu:65")
elif nnn == "8":
        #指圈
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

        # cap = cv2.VideoCapture(0)
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
                    a123 = "123"
                    start123 = time.time()
                joint_color = (0, 255, 0)  # 關節點的顏色
                line_color = (0, 0, 255)   # 骨架連接線的顏色
                joint_radius = 5         # 關節點的半徑
                line_thickness = 2         # 骨架連接線的粗細
                
        #         img = cv2.resize(img,(1500,1000))
                img = img[144:336, 192:448]
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
                cv2.namedWindow('abc123', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('abc123', 1280, 720)
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



        def cut(data):
            num_points = len(data)
            start_index = int(num_points * 0)  
            end_index = int(num_points * 1)   
            data = data[start_index:end_index]
            return data

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

        def fourier2(x, a0, a1, b1, a2, b2, w):
            return a0 + a1*np.cos(x*w) + b1*np.sin(x*w) + a2*np.cos(2*x*w) + b2*np.sin(2*x*w)
        def stadard_count(data):
            y_data = data
            num_points = len(y_data)
            start_index = int(num_points * 0.1)  
            end_index = int(num_points * 0.9)   
            y_data = y_data[start_index:end_index]
            y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))
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
                residuals_std = np.std(residuals)
                residuals_std = round(residuals_std,2)
                std_list.append(residuals_std)

            
            min_std = np.argmin(std_list) #找最小值的index
            new_guess = (min_std+1)/100

            initial_guess = [0, 0, 0, 0, 0, new_guess]  # 初始值，老師說之後要用迴圈去調整

            fit_params, _ = curve_fit(fourier2, x_data, y_data, p0=initial_guess, maxfev=5000, method='lm')

            residuals = y_data - y_fit

            residuals_std = np.std(residuals)
            residuals_std = round(residuals_std,2)

            return residuals_std

       


        g = cut(g)
        h = cut(h)


        g = -g

        std1= stadard_count(e)
        std2= stadard_count(f)

        score1 = 100 * np.exp(-(std1 - 0.13))
        score2 = 100 * np.exp(-(std2 - 0.12))
        score1 = score1 + 36
        score2 = score2 + 36
        score1 = int((np.sqrt(score1))*10)
        score2 = int((np.sqrt(score2))*10)
        if score1 >= 100:
            score1 = 100
        if score2 >= 100:
            score2 = 100




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
            score3 = 100

        # if score3 <40:
        #     score3 += 60
        # else:
        #     score3 = score3**0.5 *10
        print(f"cro:{int(score3)}")

        score4 = f_score = 100*math.exp((2-float(dotime))/2)
        score4 = score4+36
        score4 = int((np.sqrt(score4))*10)

        if score4 > 100:
            score4 = 100

        print(f"flu:{int(score4)}")

        # print(stability)
        # print(score3)
        # print(score4)
        process_file.removefile("./08/test")
        process_file.sort_and_rename_files("./save/08",video_name)
    except:
        print(f"stb:65")
        print(f"cro:65")
        print(f"flu:65")

elif nnn == "9":
    import for_score09
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



        # cap = cv2.VideoCapture(0)
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
                    a123 = "123"
                    start123 = time.time()
                joint_color = (0, 255, 0)  # 關節點的顏色
                line_color = (0, 0, 255)   # 骨架連接線的顏色
                joint_radius = 5         # 關節點的半徑
                line_thickness = 2         # 骨架連接線的粗細
                
        #         img = cv2.resize(img,(1500,1000))
                img = img[144:336, 192:448]
                image_height, image_width, _ = img.shape
                # img = img[300:image_height, 0:image_width]
                # image_height, image_width, _ = img.shape
                img = cv2.flip(img,1)
                img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
                results = hands.process(img2)                 # 偵測手掌
                
                
                set2 = []

                

                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1:
                    frame_count = frame_count+1
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
                cv2.namedWindow('abc123', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('abc123', 1280, 720)
                cv2.imshow('abc123', img)
                out.write(img)
                if cv2.waitKey(5) == ord('q'):
                    break
                if (time.time()-start123)>30:
                    break    # 按下 q 鍵停止
        elapsed_time=round(frame_count / fps,2)

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
        stability, accuracy, smooth = for_score09.process(path)
        # print(stability)
        # print(accuracy)
        # print(smooth)




        process_file.removefile("./09/test")

        process_file.sort_and_rename_files("./save/09",video_name)
    except:
        print(f"stb:65")
        print(f"cro:65")
        print(f"flu:65")
elif nnn == "12":


    try:

        sa_F = [] #存腳的距離比例成績
        sa_H = [] #存手的距離成績
        thres_F = 0.64
        thres_H = 0.45
        #下面四個是教練的四肢標準差
        LH_S_std=0.0155
        RH_S_std=0.0158
        LF_S_std=0.0255
        RF_S_std=0.0177



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
                pointL = (po17+po19+po21)/3 #手掌座標
            if   point_2 == 18 or point_2 == 20 or point_2 ==22:
                po22 = body_np[22]
                po20 = body_np[20]
                po18 = body_np[18]
                po22 = po22[1:]
                po20 = po20[1:]
                po18 = po18[1:]
                pointR = (po18+po20+po22)/3

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
            time = len(peaks)
            #print("howmany",len(peaks)) #算次數 ，踏步的開始必須是站好，不然會出問題
            
            max_vals = []
            for i in peaks:
                #print(y_data[i])
                max_vals.append(y_data[i])
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
            
            return time, peaks




        def fordist(bodynp, tall):
            global sa_F, sa_H
            first = bodynp
            tall1 = tall

            Lhip = first[23][1:]
            Rhip = first[24][1:]
            Lsh = first[11][1:]
            Rsh = first[12][1:]
            lhand,rhand = hand_point(first, 17, 18) #左右手坐標
            Lfoot,Rfoot = foot_point(first, 31, 32)

            mid_sh = (Lsh+Rsh)/2

            distL_f = np.sqrt(np.sum((Lhip-Lfoot)**2)) #腳跟肩膀的距離
            distR_f = np.sqrt(np.sum((Rhip-Rfoot)**2))
            distL_h = np.sqrt(np.sum((lhand-mid_sh)**2)) #手跟肩膀的距離
            distR_h = np.sqrt(np.sum((rhand-mid_sh)**2))


            #print("distL_f:", distL_f)
            #print("distR_f:", distR_f)
            #print("distL_h:", distL_h)
            #print("distR_h:", distR_h)
            
            if distL_f < distR_f: #表示抬左腳
                footdi = distL_f/tall1 #就是腳的比值
                sa_F.append(footdi)
            else:
                footdi = distR_f/tall1
                sa_F.append(footdi)
            #print(len(sa_F))
            if distL_h < distR_h:
                handdi = distL_h #手的距離
                sa_H.append(handdi)
            else:
                handdi = distR_h
                sa_H.append(handdi)
            #print(len(sa_H))

            return footdi, handdi


        def acc_score(foot, hand):
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

        def acc_score2(fff, hhh): #先放著，不一定用，這個就是看有幾次做到標準
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

            if len(allf) == 0 :
                totalscore = 60
            elif len(allh) == 0:
                totalscore = 60
            else:
                scoreF = int((gf/len(allf))*100)
                scoreH = int((gh/len(allh))*100)

                totalscore = int(scoreF * 0.5 + scoreH * 0.5)

                totalscore = totalscore + 36

                totalscore = int((np.sqrt(totalscore))*10)
                
                if totalscore >=100:
                    totalscore = 100

                return totalscore




        #流暢度，目前以1秒一次為標準
        def smmooth_score(time,howmany):
            elapsed_time = time
            one_time = (elapsed_time-5)/howmany
            if one_time <= 1:
                speed =100 #流暢度成績
            else:
                speed = -100*(one_time-1)+100
                speed = int(speed)+36
                if speed < 0:
                    speed = 36

            speed = int((np.sqrt(speed))*10)
            
            if speed >=100:
                speed = 100
            return speed

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic

        #一般攝影機
        #?????????
        # cap = cv2.VideoCapture(0)

        video_name = "./12/12.mp4"
        vvv = "12"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_name, fourcc, int(cap.get(cv2.CAP_PROP_FPS))/2, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        fps = cap.get(cv2.CAP_PROP_FPS)
        #print("fps:",int(cap.get(cv2.CAP_PROP_FPS)) )

        frame_count = 0
        tall=[]
        output_folder = "./"+vvv+'/test/keypoints-'+ video_name.split("/")[2].split(".")[0]  # 設定儲存關鍵點檔案的資料夾路徑
        output_folder2 = "./"+vvv+'/test/keypoints-3d-'+ video_name.split("/")[2].split(".")[0]
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(output_folder2, exist_ok=True)

        a123 = ""
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            start_time = time.time() #設定開始時間，這個動作可以給3秒倒數計時


            allkeys = np.empty((0, 4), dtype=np.float32)
            while True:
                
                ret, img = cap.read()
                img= cv2.flip(img,1)
                h, w, c = img.shape
                if not ret:
                    #print("Cannot receive frame")
                    break
                if a123 == "": #用來擋，卡在第一幀不動，有輸入才會繼續錄
                    a123 = "123"
                    start123 = time.time()
                results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                if results.pose_landmarks:
                    frame_count = frame_count+1
                    #print("frame count:",frame_count)
                    use_keys = np.empty((0, 4), dtype=np.float32)
                    key_dist = np.empty((0, 4), dtype=np.float32) #用來算距離用
                    all3dkeys = np.empty((0, 4), dtype=np.float32)
                    mp_drawing.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
                    
                    for id, landmark in enumerate(results.pose_landmarks.landmark):

                        use_key = np.array([id,landmark.x, landmark.y, landmark.z], dtype=np.float32)
                        use_keys = np.vstack([use_keys, use_key])
                        
                        
                        key_dist= np.vstack([key_dist, use_key])
                        output_file = os.path.join(output_folder, f'keypoints_{frame_count}.npy')
                        np.save(output_file, key_dist)
                        #print("put 1")
                    for id, landmark in enumerate(results.pose_world_landmarks.landmark):
                        use_key = np.array([id,landmark.x, landmark.y, landmark.z], dtype=np.float32)
                        allkeys = np.vstack([allkeys, use_key])
                        all3dkeys = np.vstack([all3dkeys, use_key]) 
                        
                        output_file = os.path.join(output_folder2, f'keypoints_{frame_count}.npy')
                        np.save(output_file, all3dkeys)
                        #print("put 2")
                    if frame_count<31:
                        LSHdis = np.sqrt(np.sum((all3dkeys[23][1:]-all3dkeys[31][1:])**2))
                        tall.append(LSHdis)
                    tall__ = sum(tall)/len(tall)
                    #print("tall:",tall__)
                    
                    leg, handsss = fordist(all3dkeys,tall__)
                    

                    

                    
                    

                    #cv2.putText(img,("time:"+str(round((time.time() - start_time),2))), (40, 60), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (0,255,0),2)
                    #cv2.putText(img,("hand:"+str(round(handsss,2))), (40, 80), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (200,0,255),2)
                    #cv2.putText(img,("foot:"+str(round(leg,2))), (40, 100), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (200,0,255),2)




                cv2.namedWindow('abc123', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('abc123', 1280, 720)
                cv2.imshow('abc123', img)
                out.write(img)
                # key = cv2.waitKey(1)
                if cv2.waitKey(5) == ord('q'):
                    break
                if (time.time() -  start123) >30:
                    break

        cv2.destroyAllWindows()
        cap.release()
        out.release()


        elapsed_time=round(frame_count / fps,2)




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
        howmany, where = stad_count(path3) #算總共做了幾次
        #print("times:",howmany)
        #print("where:",where)
        #print("ffff:",len(sa_F))
        #print("hhhh:",len(sa_H))
        get_F = []
        get_H = []
        for i in where: #找到波峰的位置，把那個點的值存下來
            if len(sa_H)>i and len(sa_F)>i:
                get_F.append(sa_F[i])
                get_H.append(sa_H[i])

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

        LH, RH, LF, RF = for_step.step_top (output_folder2,howmany) #這次動作的四肢std
        LH_score =calculate_std(LH,LH_S_std)#四個成績
        RH_score =calculate_std(RH,RH_S_std)
        LF_score =calculate_std(LF,LF_S_std)
        RF_score =calculate_std(RF,RF_S_std)
        std_score3 = int((LH_score+RH_score+LF_score+RF_score)/4)
        std_score3 = std_score3+36
        std_score3 = int((np.sqrt(std_score3))*10)
        if std_score3 >=100:
            std_score3 = 100


        #print("LH_score:",LH_score)
        #print("RH_score:",RH_score)
        #print("LF_score:",LF_score)
        #print("RF_score:",RF_score)



        #print(f"穩定度評分3:{std_score3}分")
        #print(f"準確度評分:{accuracy}分")
        #print(f"流暢度評分3:{speed}分")
        if std_score3 < 20:
            std_score3 += 60
        if accuracy < 20:
            accuracy += 60
        if speed < 20:
            speed += 60
        print(f"stb:{std_score3}")
        print(f"cro:{accuracy}")
        print(f"flu:{speed}")
                    

                    # img12 = np.zeros((256, 256, 3), np.uint8)

                    # img12 = cv2.resize(img12,(700,700))

                    # # 將圖片用淺灰色 (200, 200, 200) 填滿
                    # img12.fill(255)

                    # # 在圖片上畫一條紅色的對角線，寬度為 5 px
                    # cv2.putText(img12, ("Stability score: " + str(std_score3)), (10, 50), cv2.FONT_HERSHEY_DUPLEX,
                    # 1, (0, 0, 0), 1, cv2.LINE_AA)

                    # cv2.putText(img12, ("Accuracy score: " + str(accuracy)), (10, 100), cv2.FONT_HERSHEY_DUPLEX,
                    # 1, (0, 0, 0), 1, cv2.LINE_AA)

                    # cv2.putText(img12, ("Smoothness score: " + str(speed)), (10, 150), cv2.FONT_HERSHEY_DUPLEX,
                    # 1, (0, 0, 0), 1, cv2.LINE_AA)


                    # # 顯示圖片
                    # cv2.imshow('My Image', img12)

                    # 按下任意鍵則關閉所有視窗




                




        process_file.removefile("./12/test")
        process_file.sort_and_rename_files("./save/12",video_name)


    except:
        print(f"stb:65")
        print(f"cro:65")
        print(f"flu:65")
elif nnn == "14":
    import for_touch

    try:
        #手摸腳後跟的起始動作不是原地站好，是先摸一邊

        #touch_dist = 0.5 #教練手碰腳跟距離閾值
        touch_dist = 0.5 #教練手碰腳跟距離閾值
        LH_B_std = 0.0442 
        RH_B_std = 0.0134
        LH_S_std = 0.0247
        RH_S_std = 0.0253
        LF_S_std = 0.0148
        RF_S_std = 0.0183

        sa_F = [] #存腳的距離比例成績
        sa_H = [] #存手的距離成雞




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
                pointL = (po17+po19)/2 #手掌座標
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

            peaks, _ = find_peaks(y_data, prominence=0.02, distance =15)
            valleys, _ = find_peaks(-y_data, prominence=0.02, distance =15)
            #print("peak:",peaks)
            time = len(peaks)+len(valleys) 
            #print("howmany",time) #算次數 ，碰腳跟的開始是一邊已經碰到了，所以要peak跟valley次數加起來
            
            max_vals = []
            min_vals = []
            for i in peaks:
                #print(y_data[i])
                max_vals.append(y_data[i])
            for j in valleys:
                #print(y_data[i])
                min_vals.append(y_data[j])

            
            #print('Max values:', max_vals)
            #print('Min values:', min_vals)


            #plt.plot(x_fit, y_fit, 'r-', label='Fit')
            plt.plot(y_data)
            plt.xlabel('x')
            plt.ylabel('y')
            x_major_locator = MultipleLocator(10) #把x軸刻度設1, 存在變數裡
            ax = plt.gca() #ax為兩個座標軸的實例
            ax.xaxis.set_major_locator(x_major_locator) #把x座標軸刻度設為1的倍數
            plt.xlim(0.5,len(y_data))
            plt.scatter(peaks, max_vals, c ='red')
            plt.scatter(valleys, min_vals, c ='green')
            
            # 顯示圖表
            pic_num = csv_name.split(".")[1].split('/')[1]
            #print(pic_num)
            cccc = pic_num
            folder = "./"+cccc+"/test/"
            pic = folder +pic_num +"_up.png" #圖片存檔
            plt.savefig(pic)

            return time, peaks , valleys

        def fordist(bodynp):
            global sa_F
            first = bodynp

            lhand,rhand = hand_point(first, 17, 18) #左右手坐標
            # Lfoot,Rfoot = foot_point(first, 31, 32)
            Lfoot = first[29][1:]
            Rfoot = first[30][1:]

            distL_f = np.sqrt(np.sum((rhand-Lfoot)**2)) #腳跟手掌的距離
            distR_f = np.sqrt(np.sum((lhand-Rfoot)**2))



            #print("distL_f:", distL_f)
            #print("distR_f:", distR_f)
            
            if distL_f < distR_f: #表示抬左腳
                footdi = distL_f
                sa_F.append(footdi)
            else:
                footdi = distR_f
                sa_F.append(footdi)
            
            
            return footdi

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


        def acc_score(foot ,touch_dist): #touch_dist目前是0.45
            #目前0.45是有碰到
            if foot <= touch_dist:
                scoreF = 100
            else:
                scoreF = -300 * (foot - touch_dist) + 100
            
            #print("scoreF:",scoreF)
            # if hand <= 0.6:
            #     scoreH = 100
            # else:
            #     scoreH = -250 * (hand - 0.6) + 100
            # print("scoreH:",scoreH)

            # totalscore = int(scoreF * 0.6 + scoreH * 0.4)
            totalscore = int(scoreF)

            return  totalscore

        def acc_score2(fff): #先放著，不一定用，這個就是看有幾次做到標準
            global touch_dist
            allf = fff
            gf = 0
            for i in allf:
                #print(i)
                if i <= touch_dist:
                    gf+=1
            #print("good:",gf)

            scoreF = int((gf/len(allf))*100)

            totalscore = scoreF + 36
            totalscore = int((np.sqrt(totalscore))*10)
                
            if totalscore >=100:
                totalscore = 100

            return totalscore


        #流暢度，目前以1.1秒一次為標準
        def smmooth_score(time , howmany): #總時間跟次數
            elapsed_time = time
            one_time = (elapsed_time-5)/howmany
            if one_time <= 1.1:
                speed =100 #流暢度成績
            else:
                speed = -100*(one_time-1.1)+100
                speed = int(speed)+36
                if speed < 0:
                    speed = 36

            speed = int((np.sqrt(speed))*10)
            
            if speed >=100:
                speed = 100
            return speed



        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_holistic = mp.solutions.holistic

        #一般攝影機
        # cap = cv2.VideoCapture(0)

        video_name = "./14/14.mp4"
        vvv = "14"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_name, fourcc, int(cap.get(cv2.CAP_PROP_FPS))/2, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        output_folder = "./"+vvv+'/test/keypoints-'+ video_name.split("/")[2].split(".")[0]  # 設定儲存關鍵點檔案的資料夾路徑
        output_folder2 = "./"+vvv+'/test/keypoints-3d-'+ video_name.split("/")[2].split(".")[0]
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(output_folder2, exist_ok=True)

        a123 = ""
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            start_time = time.time() #設定開始時間，這個動作可以給3秒倒數計時


            allkeys = np.empty((0, 4), dtype=np.float32)
            while True:
                
                ret, img = cap.read()
                img= cv2.flip(img,1)
                h, w, c = img.shape
                if not ret:
                    #print("Cannot receive frame")
                    break
                if a123 == "": #用來擋，卡在第一幀不動，有輸入才會繼續錄
                    a123 ="123"
                    start123 = time.time()
                results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    frame_count = frame_count+1
                    #print("frame count:",frame_count)
                    use_keys = np.empty((0, 4), dtype=np.float32)
                    key_dist = np.empty((0, 4), dtype=np.float32) #用來算距離用
                    all3dkeys = np.empty((0, 4), dtype=np.float32)
                    mp_drawing.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
                    
                    for id, landmark in enumerate(results.pose_landmarks.landmark):

                        use_key = np.array([id,landmark.x, landmark.y, landmark.z], dtype=np.float32)
                        use_keys = np.vstack([use_keys, use_key])
                        allkeys = np.vstack([allkeys, use_key])
                        
                        key_dist= np.vstack([key_dist, use_key])
                        output_file = os.path.join(output_folder, f'keypoints_{frame_count}.npy')
                        np.save(output_file, key_dist)
                        #print("put 1")
                    for id, landmark in enumerate(results.pose_world_landmarks.landmark):
                        use_key = np.array([id,landmark.x, landmark.y, landmark.z], dtype=np.float32)
                        
                        all3dkeys = np.vstack([all3dkeys, use_key]) 
                        output_file = os.path.join(output_folder2, f'keypoints_{frame_count}.npy')
                        np.save(output_file, all3dkeys)
                        #print("put 2")

                    leg = fordist(all3dkeys)

                    #cv2.putText(img,("time:"+str(round((time.time() - start_time),2))), (40, 60), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (0,255,0),2)
                    #cv2.putText(img,("foot:"+str(round(leg,2))), (40, 100), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (200,0,255),2)





                cv2.namedWindow('abc123', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('abc123', 1280, 720)
                cv2.imshow('abc123', img)
                out.write(img)
                # key = cv2.waitKey(1)
                if cv2.waitKey(5) == ord('q'):
                    break
                if (time.time() - start123)>30:
                    break


        cap.release()
        out.release()
        cv2.destroyAllWindows()
        elapsed_time = round(frame_count / fps,2)



        np_path = [os.path.join(output_folder,f) for f in os.listdir(output_folder)if f.endswith('.npy')]           
        np_sorted = natsorted (np_path) 
        dist_all=[]
        for i, file in enumerate(np_sorted):
            get=np.load(file)
            np0 = np.load(np_sorted[0])
            #print(get.shape)
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
            #print(get.shape)
            if get.shape == np0.shape:
                #print(file)
                ske_dist = euclidean.eucliDist(np_sorted3[0], file)
                #ske_dist = euclidean.eucliDist("./yn01/pic/yn01brain/np/frame0_key.npy", file)
                dist_all3.append(ske_dist)
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

        #print("len:", len(sa_F))
        #開始算成績
        howmany, whpeak, whvalley = stad_count(path3) #算總共做了幾次
        #print("times:",howmany)
        #print("whpeak:",whpeak)
        #print("whvalley:",whvalley)
        get_F = []
        # for i in whpeak: #找到波峰的位置，把那個點的值存下來
        #     if len(sa_F)>i:
                
        #         get_F.append(sa_F[i])
        # for i in whvalley:#找到波谷的位置，把那個點的值存下來
        #     if len(sa_F)>i:
        #         get_F.append(sa_F[i])



        #這邊是要找波峰和波谷的前2幀跟後2幀的值，取最小的存

        peakva = catch2_2(whpeak,sa_F)
        valleyva = catch2_2(whvalley,sa_F)
        peakva.extend(valleyva) #把兩邊的值加起來
        get_F = peakva


            

        #print("get_F:",get_F)
        #avg_F = sum(get_F)/len(get_F) #平均的腳
        #accuracy = acc_score(avg_F,touch_dist) #準確度成績
        accuracy = acc_score2(get_F) #另一種


        speed = smmooth_score(elapsed_time,howmany) #流暢度成績

        #標準差
        # this_std3 = Std_of_stability.stadard_count(path3)#3d的標準差
        # coach_std3 = Std_of_stability.stadard_count('./csv/12.csv') #教練
        # std_score3 = int(calculate_std(this_std3,coach_std3))


        LH, RH, LF, RF = for_touch.step_top (output_folder2,howmany)
        #14要用的是下面的手，抬起的腳，所以是手是y大的，腳是y小的
        LH_score =calculate_std(LH,LH_B_std)#四個成績
        RH_score =calculate_std(RH,RH_B_std)
        LF_score =calculate_std(LF,LF_S_std)
        RF_score =calculate_std(RF,RF_S_std)
        std_score3 = int((LH_score+RH_score+LF_score+RF_score)/4)
        std_score3 = std_score3+36
        std_score3 = int((np.sqrt(std_score3))*10)
        if std_score3 >=100:
            std_score3 = 100



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



        # img12 = np.zeros((256, 256, 3), np.uint8)

        # img12 = cv2.resize(img12,(700,700))

        # # 將圖片用淺灰色 (200, 200, 200) 填滿
        # img12.fill(255)

        # # 在圖片上畫一條紅色的對角線，寬度為 5 px
        # cv2.putText(img12, ("Stability score: " + str(std_score3)), (10, 50), cv2.FONT_HERSHEY_DUPLEX,
        # 1, (0, 0, 0), 1, cv2.LINE_AA)

        # cv2.putText(img12, ("Accuracy score: " + str(accuracy)), (10, 100), cv2.FONT_HERSHEY_DUPLEX,
        # 1, (0, 0, 0), 1, cv2.LINE_AA)

        # cv2.putText(img12, ("Smoothness score: " + str(speed)), (10, 150), cv2.FONT_HERSHEY_DUPLEX,
        # 1, (0, 0, 0), 1, cv2.LINE_AA)


        # # 顯示圖片
        # cv2.imshow('My Image', img12)

        # 按下任意鍵則關閉所有視窗




                





        process_file.removefile("./14/test")
        process_file.sort_and_rename_files("./save/14",video_name)

    except:
        print(f"stb:65")
        print(f"cro:65")
        print(f"flu:65")
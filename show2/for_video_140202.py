import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from PIL import Image
from sympy import*
import math
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
import for_touch


#手摸腳後跟的起始動作不是原地站好，是先摸一邊

#touch_dist = 0.5 #教練手碰腳跟距離閾值
touch_dist = 0.5 #教練手碰腳跟距離閾值
LH_B_std = 0.0384 
RH_B_std = 0.0149
LH_S_std = 0.0243
RH_S_std = 0.0256
LF_S_std = 0.014
RF_S_std = 0.0188

sa_F = [] #存腳的距離比例成績
sa_H = [] #存手的距離成雞
pose_2d=[]
pose_3d=[]


def count_time(): #算時間
    global do_time,num_of_time
    do_time.append(time.time())
    if num_of_time > 0:
        print(f"time{num_of_time}:{do_time[num_of_time] - do_time[num_of_time-1]}")
    num_of_time += 1


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
    print("min10dist:",sorted_list[:n])
    print("min10:",bottom_10_index)
    return bottom_10_index ,sorted_list[:n]



def fourier2(x, a0, a1, b1, a2, b2, w):
    return a0 + a1*np.cos(x*w) + b1*np.sin(x*w) + a2*np.cos(2*x*w) + b2*np.sin(2*x*w)

def stad_count(csv_name,lis): #算次數用


    numpy_array = np.array(lis, dtype=float) #把距離都取出來
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
    
   
    print("参数:", fit_params)



    # 畫圖
    #plt.scatter(x_data, y_data, label='Data') #原本的距離跟len


    x_fit = x_data
    y_fit = fourier2(x_data, *fit_params)

    peaks, _ = find_peaks(y_data, prominence=0.1, distance =20)
    valleys, _ = find_peaks(-y_data, prominence=0.1, distance =20)
    print("peak:",peaks)
    time = len(peaks)+len(valleys) 
    print("howmany",time) #算次數 ，碰腳跟的開始是一邊已經碰到了，所以要peak跟valley次數加起來
    
    max_vals = []
    min_vals = []
    for i in peaks:
        #print(y_data[i])
        max_vals.append(y_data[i])
    for j in valleys:
        #print(y_data[i])
        min_vals.append(y_data[j])

    
    print('Max values:', max_vals)
    print('Min values:', min_vals)


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


    pic_num = csv_name
    print(pic_num)
    cccc = pic_num
    folder = "./"+cccc+"/test/"
    pic = folder +pic_num +"_up.png" #圖片存檔
    plt.savefig(pic)
    #plt.show()

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



    print("distL_f:", distL_f)
    print("distR_f:", distR_f)
    
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
            print(i)
            find.append(i-2) #開始存前後2幀的index
            find.append(i-1)
            find.append(i)
            find.append(i+1)
            find.append(i+2)
        print(find)
        for i in find: #拿掉不能用的index
            #print(i)
            if i<0 or i>=len(lis):
                find.remove(i)
        print("find:",find)
        value_ = [] #用來放這五幀的值
        for i in find:
            print("gp:",lis[i])
            value_.append(lis[i])
        print("sum:",value_)
        go = min(value_)
        get.append(go)#取最小的存
        print("min:",go)

    return get


def acc_score(foot ,touch_dist): #touch_dist目前是0.45
    #目前0.45是有碰到
    if foot <= touch_dist:
        scoreF = 100
    else:
        scoreF = -300 * (foot - touch_dist) + 100
    
    print("scoreF:",scoreF)
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
        print(i)
        if i <= touch_dist:
            gf+=1
    print("good:",gf)

    scoreF = int((gf/len(allf))*100)

    totalscore = scoreF

    return totalscore


#流暢度，目前以1.1秒一次為標準
def smmooth_score(time , howmany): #總時間跟次數
    elapsed_time = time
    one_time = (elapsed_time-5)/howmany
    if one_time <= 1.1:
        speed =100 #流暢度成績
    else:
        speed = -100*(one_time-1.1)+100
        speed = int(speed)
        if speed < 0:
            speed = 0
    return speed

def for_video14(readvideo):

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic



    #輸入的影片
    vvv = readvideo
    path = "./"+vvv+"/"+vvv+".mp4"
    new_path = "./"+vvv+"/test/"+vvv+"-score.avi"
    score_path = "./"+ vvv+"/test/"+vvv+"-score.png"
    cap = cv2.VideoCapture(path)
    print(int(cap.get(cv2.CAP_PROP_FPS)))

    video_name = new_path #要存成什麼
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    allframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_name, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    output_folder = "./"+vvv+'/test/keypoints-'+ video_name.split("/")[2].split(".")[0]  # 設定儲存關鍵點檔案的資料夾路徑
    output_folder2 = "./"+vvv+'/test/keypoints-3d-'+ video_name.split("/")[2].split(".")[0]
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)


    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        start_time = time.time() #設定開始時間，這個動作可以給3秒倒數計時


        allkeys = np.empty((0, 4), dtype=np.float32)
        while True:

            ret, img = cap.read()
            print(ret)
            if not ret:
                break
            results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                frame_count = frame_count+1
                print("frame count:",frame_count)
                use_keys = np.empty((0, 4), dtype=np.float32)
                key_dist = np.empty((0, 4), dtype=np.float32) #用來算距離用
                all3dkeys = np.empty((0, 4), dtype=np.float32)
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                   .get_default_pose_landmarks_style())
                h, w, c = img.shape
                for id, landmark in enumerate(results.pose_landmarks.landmark):

                    use_key = np.array([id,landmark.x, landmark.y, landmark.z], dtype=np.float32)
                    use_keys = np.vstack([use_keys, use_key])
                    allkeys = np.vstack([allkeys, use_key])

                    key_dist= np.vstack([key_dist, use_key])
                    #output_file = os.path.join(output_folder, f'keypoints_{frame_count}.npy')
                    #np.save(output_file, key_dist)
                    #print("put 1")
                pose_2d.append(key_dist)
                for id, landmark in enumerate(results.pose_world_landmarks.landmark):
                    use_key = np.array([id,landmark.x, landmark.y, landmark.z], dtype=np.float32)

                    all3dkeys = np.vstack([all3dkeys, use_key]) 
                    #output_file = os.path.join(output_folder2, f'keypoints_{frame_count}.npy')
                    #np.save(output_file, all3dkeys)
                    #print("put 2")
                pose_3d.append(all3dkeys)

                leg = fordist(all3dkeys)

                
                #cv2.putText(img,("foot:"+str(round(leg,2))), (40, 100), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (200,0,255),2)





            #cv2.namedWindow('RGB Example', cv2.WINDOW_NORMAL)
            #cv2.resizeWindow('RGB Example', frame_width, frame_height)
            #cv2.imshow('RGB Example', img)
            #out.write(img)

    #         if cv2.waitKey(1) & 0xFF == ord('q'):

    #             break

        # 計算經過的時間
    elapsed_time1 = time.time() - start_time
    print("elapsed_time1",elapsed_time1)
#     elapsed_time = round(elapsed_time,2)

    elapsed_time = round(frame_count / fps,2)
    

    #np_path = [os.path.join(output_folder,f) for f in os.listdir(output_folder)if f.endswith('.npy')]           
    #np_sorted = natsorted (np_path) 
    dist_all=[]

    for i in range(len(pose_2d)):
        #get=np.load(file)
        #np0 = np.load(np_sorted[0])
        #rint(get.shape)
        ske_dist = np.sqrt(np.sum((pose_2d[0]-pose_2d[i])**2))
        #ske_dist = euclidean.eucliDist("./yn01/pic/yn01brain/np/frame0_key.npy", file)
        dist_all.append(ske_dist)
    #print("all:",dist_all)
    print("lendist:",len(dist_all))
    
    
    # headers = ["pic", "dist"]
    # csv_name = "./"+vvv+"/test/"+video_name.split("/")[2].split(".")[0]
    # path = csv_name + ".csv"
    # with open(path, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(headers)
    #     for i in range(len(dist_all)):
    #         writer.writerow([i,dist_all[i]]) #存影片的動作所有距離


    #3D        
    #np_path3 = [os.path.join(output_folder2,f) for f in os.listdir(output_folder2)if f.endswith('.npy')]           
    #np_sorted3 = natsorted (np_path3) 
    dist_all3=[]
    for i in range(len(pose_3d)):

        #ske_dist3 = euclidean.eucliDist_no(np_sorted[0], file, 11 ,32)
        ske_dist3= np.sqrt(np.sum((pose_3d[0]-pose_3d[i])**2))
        #ske_dist = euclidean.eucliDist("./yn01/pic/yn01brain/np/frame0_key.npy", file)
        dist_all3.append(ske_dist3)
    #print("all:",dist_all3)
    #print("lendist:",len(dist_all3))

    #print("len:", len(sa_F))
    #開始算成績
    howmany, whpeak, whvalley = stad_count(vvv,dist_all3) #算總共做了幾次
    print("times:",howmany)
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


    print("LH_score:",LH_score)
    print("RH_score:",RH_score)
    print("LF_score:",LF_score)
    print("RF_score:",RF_score)
    print("fps:",fps)
    print("all:",allframe)
    print("frame:",frame_count)
    print("時間:",elapsed_time)

    print(f"穩定度評分3:{std_score3}分")
    print(f"準確度評分:{accuracy}分")
    print(f"流暢度評分3:{speed}分")
    


    cv2.destroyAllWindows()
    img12 = np.zeros((256, 256, 3), np.uint8)

    img12 = cv2.resize(img12,(700,700))

    # 將圖片用淺灰色 (200, 200, 200) 填滿
    img12.fill(255)

    # 在圖片上畫一條紅色的對角線，寬度為 5 px
    cv2.putText(img12, ("Stability score: " + str(std_score3)), (10, 50), cv2.FONT_HERSHEY_DUPLEX,
    1, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(img12, ("Accuracy score: " + str(accuracy)), (10, 100), cv2.FONT_HERSHEY_DUPLEX,
    1, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(img12, ("Smoothness score: " + str(speed)), (10, 150), cv2.FONT_HERSHEY_DUPLEX,
    1, (0, 0, 0), 1, cv2.LINE_AA)


    # 顯示圖片
    #cv2.imshow('My Image', img12)
    cv2.imwrite(score_path,img12)

    # 按下任意鍵則關閉所有視窗

    cv2.destroyAllWindows()


    # 釋放影片物件
    cap.release()
    out.release()

    # 關閉視窗
    cv2.destroyAllWindows()

    return std_score3, accuracy ,speed
    
#for_video14("14")
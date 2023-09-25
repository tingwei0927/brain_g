import numpy as np
import math
import os
from natsort import natsorted
import cv2

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
    if  point_2 == 18 or point_2 == 20 or point_2 ==22:
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


def get_minandmax( popo, k,n=10): #要用的nparray, 要取幾個值, 要用哪個比大小
    
    sorted_indices = np.argsort(popo[:, k])  # 根据第二列进行排序
    
    # 获取最大的n个值及其索引
    top_32_max_indices = sorted_indices[-n:]
    top_32_max_values = popo[top_32_max_indices, k]
    top_32_max_positions = popo[top_32_max_indices, :]
    
    
    # 获取最小的n个值及其索引
    top_32_min_indices = sorted_indices[:n]
    top_32_min_values = popo[top_32_min_indices, k]
    top_32_min_positions = popo[top_32_min_indices, :]#全部
    
    #print("Top 32 max values:", top_32_max_values)
    #print("top_32_max_indices:", top_32_max_indices)
    
    #print("Top 32 min values:", top_32_min_values)
    #print("top_32_min_indices:", top_32_min_indices)
    
    return top_32_max_indices, top_32_max_positions, top_32_min_indices , top_32_min_positions


def count_std(goarray):#算標準差

    all = goarray
    std_ = np.std(all, axis = 0)
    print("std",std_)
    #print("std_",std_)
    avg_std= round(sum(std_)/len(std_), 4)
    #print("avg:",avg_std)
    return avg_std
    
            


def step_top(folderpath, num): #要測的3d nparray資料夾，要找幾個最高點(因為老師說先分別找四肢的最高點就好
    folder_path = folderpath
    np_path = [os.path.join(folder_path,f) for f in os.listdir(folder_path)if f.endswith('.npy')]
    np_sorted = natsorted (np_path)
    po_LH = np.empty((0, 3), dtype=np.float32)
    po_RH = np.empty((0, 3), dtype=np.float32)
    po_LF = np.empty((0, 3), dtype=np.float32)
    po_RF = np.empty((0, 3), dtype=np.float32)

    
    for i, file in enumerate(np_sorted):
        #print("fff:",file)
        this = np.load(file)
        lhand, rhand = hand_point(this, 17 , 18)
        #print("LH",lhand)
        #print("RH",rhand)
        lfoot, rfoot = foot_point(this, 31 , 32)
        #print("lfoot",lfoot)
        po_LH =np.vstack([po_LH, lhand])
        po_RH =np.vstack([po_RH, rhand])
        po_LF =np.vstack([po_LF, lfoot])
        po_RF =np.vstack([po_RF, rfoot])
        #print("-----------------")
    #print(len(po_RH))
    
    #print(type(po_RH))
    
    #要比較y值
    bigLH, bbbLH, smaLH, sssLH = get_minandmax(po_LH,1, num)#手的最低點位置跟值，和最高點位置跟值(因為y是越往下越負)
    bigRH, bbbRH, smaRH, sssRH = get_minandmax(po_RH,1, num)
    bigLF, bbbLF, smaLF, sssLF = get_minandmax(po_LF,1, num)#腳的數值大的是在地上，小的才是抬高
    bigRF, bbbRF, smaRF, sssRF = get_minandmax(po_RF,1, num)
    #LH_B_std =count_std(bbbLH,32)
    LH_S_std =count_std(sssLH)
    #RH_B_std =count_std(bbbRH,32)
    RH_S_std =count_std(sssRH)
    #LF_B_std =count_std(bbbLF,32)
    LF_S_std =count_std(sssLF)
    #RF_B_std =count_std(bbbRF,32)
    RF_S_std =count_std(sssRF)
    #都先拿小的，因為手是小的才是有高，然後腳是抬越高越小(越靠近hip)
    print("LH_S_std",LH_S_std)
    print("RH_S_std",RH_S_std)
    print("LF_S_std",LF_S_std)
    print("RF_S_std",RF_S_std)
    return LH_S_std ,RH_S_std ,LF_S_std ,RF_S_std
    
    
    
    
    

#LH, RH, LF, RF =step_top("./12/pic/12/np",32)

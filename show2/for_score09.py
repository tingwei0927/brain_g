import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import math
import emotion
import random


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


def draw(data1,peak):
    x_data = np.arange(1, len(data1)+1)
    plt.plot(x_data, data1, label='Data')
    for i in range(len(peak)):
        plt.scatter(peak[i], data1[peak[i]])



    # plt.plot(x_fit, y_fit, 'r-', label='Fit')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    #plt.show()

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
    
   
    

    #print("参数:", fit_params)



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
    #print("go",(np.sum(np.abs(residuals)**2)/(len(residuals)-1))**(1/2))
    #print("N:", str(len(residuals)))
    #print("dev:", np.sum(y_data))
    residuals_std = np.std(residuals)
    residuals_std = round(residuals_std,2)
    #print("標準差:", residuals_std)
    return residuals_std


def process(csvpath):    
    # Open CSV
    #with open('01brainpic.csv', 'r') as file:
    #with open('04.csv', 'r') as file:
    a_data = []
    b_data = []
    c_data = []
    d_data = []
    with open(csvpath, 'r') as csvfile:
        # read csv
        rows = csv.reader(csvfile)



        # 以迴圈輸出每一列
        for row in rows:
            a_data.append(float(row[1]))
            b_data.append(float(row[2]))
            c_data.append(float(row[3]))
            dotime = (row[4])



    a_data = np.array(a_data, dtype=float)
    b_data = np.array(b_data, dtype=float)
    stability_1 = a_data*1000
    stability_2 = b_data*1000



    a_data = box_smooth(a_data, 10)
    b_data = box_smooth(b_data, 10)
    c_data = box_smooth(c_data, 10)
    a_data = np.array(a_data, dtype=float)
    b_data = np.array(b_data, dtype=float)
    c_data = np.array(c_data, dtype=float)

    a_data = -a_data
    b_data = -b_data






    # draw(b_data)
    # draw(c_data)


    peaks_b, _ = find_peaks(a_data,distance=10,prominence=0.0002)
    peaks_a, _ = find_peaks(b_data,distance=10,prominence=0.0002)

    draw(a_data,peaks_a)
    draw(b_data,peaks_b)
    draw(c_data,[])
    score = 0
    torf = []
    for i in range(len(peaks_a)):
        if c_data[peaks_a[i]] < 0:
            torf.append(-1)
        else:
            torf.append(1)
        tmp = 0
        for n in range(len(peaks_b)):
            if peaks_a[i]-10 < peaks_b[n] <peaks_a[i]+10:
                tmp = 1
            if i > 1 :
                if torf[i] == torf[i-1]:
                    tmp = 0

        score += tmp

    score3 = 100*(score/len(peaks_a))
    score3 = int(score3)+36
    score3 = int((np.sqrt(score3))*10)
    if score3 >= 100:
        score3 = random.randint(91, 100)



    std1 = stadard_count(stability_1)
    std2 = stadard_count(stability_2)

    score1 = 100 * np.exp(-(std1 - 0.11))
    score2 = 100 * np.exp(-(std2 - 0.14))

    score1 = score1 + 36
    score2 = score2 + 36
    score1 = int((np.sqrt(score1))*10)
    score2 = int((np.sqrt(score2))*10)

    if score1 >= 100:
        score1 = random.randint(91, 100)
    if score2 >= 100:
        score2 = random.randint(91, 100)


   # print("do",dotime)
    one_time = float(dotime)/len(peaks_a)
    score4 = f_score = 100*math.exp((2-float(one_time))/2)
    score4 = int(score4)+36
    score4 = int((np.sqrt(score4))*10)
    

    if score4 > 100:
        score4 = random.randint(91, 100)



    print(f"stb:{int((score1 + score2)/2)}")
    sta = int((score1 + score2)/2)
    # if score3 <20:
    #     score3 += 60
    print(f"cro:{int(score3)}")
    print(f"flu:{int(score4)}")
    emoscore = emotion.emotionscore()*0.7
    emoscore = int(emoscore + (sta + score3 + score4 )/3*0.3)

    print(f"emo:{emoscore}")

    return sta, int(score3), int(score4)
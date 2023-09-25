
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit





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
    # print("go",(np.sum(np.abs(residuals)**2)/(len(residuals)-1))**(1/2))
    #print("N:", str(len(residuals)))
    #print("dev:", np.sum(y_data))
    residuals_std = np.std(residuals)
    residuals_std = round(residuals_std,2)
    # print("標準差:", residuals_std)
    return residuals_std
# Open CSV
#with open('01brainpic.csv', 'r') as file:
#with open('04.csv', 'r') as file:




def process(csvpath):
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
            d_data.append(float(row[4]))
            dotime = (row[5])
        



        # print(a_array)
        # print(a_data)



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
    sta = int((score1 + score2)/2)
    print(f"stb:{int((score1 + score2)/2)}")
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

    print(f"cro:{int(score3)}")

    score4 = f_score = 100*math.exp((2-float(dotime))/2)
    if score4 > 100:
        score4 = 100



    #print(f"流暢度:94")
    print(f"flu:{int(score4)}")
    
    return int(sta), int(score3), int(score4)

 
#stability, accuracy, smooth = process("C:\\Users\\user\\Desktop\\hand_allf.csv")
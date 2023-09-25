

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Open CSV
#with open('01brainpic.csv', 'r') as file:
#with open('04.csv', 'r') as file:

# fitting function
def fourier2(x, a0, a1, b1, a2, b2, w):
    return a0 + a1*np.cos(x*w) + b1*np.sin(x*w) + a2*np.cos(2*x*w) + b2*np.sin(2*x*w)


def stadard_count(csv_name):
    with open(csv_name, 'r') as file:
        # read csv
        reader = csv.reader(file)
        column2_data = [row[1] for row in reader]

    numpy_array = np.array(column2_data[1:], dtype=float)
    #print(numpy_array)


    



    # 取中間80%資料
    y_data = numpy_array
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
    
   
    

    print("参数:", fit_params)



    # 畫圖
    plt.scatter(x_data, y_data, label='Data')
    x_fit = x_data
    y_fit = fourier2(x_data, *fit_params)

    plt.plot(x_fit, y_fit, 'r-', label='Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


    residuals = y_data - y_fit
    print("go",(np.sum(np.abs(residuals)**2)/(len(residuals)-1))**(1/2))
    #print("N:", str(len(residuals)))
    #print("dev:", np.sum(y_data))
    residuals_std = np.std(residuals)
    residuals_std = round(residuals_std,2)
    print("標準差:", residuals_std)
    return residuals_std

#std1= stadard_count('./csv/608-117.csv') #2d
#std2= stadard_count('./csv/01brainpic-2.csv')# 2d
#std3= stadard_count('./csv/12.csv')
#std4= stadard_count('./csv/612-12-8-1.csv')
#std5= stadard_count('./csv/yn12.csv')


#std3 = stadard_count('./csv/608-117-1.csv') #3d
#std4= stadard_count('01brainpic.csv') #3d
# score = 100 * np.exp(-(std1 - std2)/0.11)+20
# score1 = 100 * np.exp(-(std3 - std4)/0.11)+20
# #score1 = 100 * np.exp(-(0.06)/0.11)+20
# score2 = 100 * np.exp(-(0.09)/0.11)+20
# print("score",score)
# print("score1",score1)
# print("score2",score2)






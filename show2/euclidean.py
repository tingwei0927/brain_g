import math
import numpy as np
import os

def eucliDist(A, B):
    first = np.load(A)
    second = np.load(B)
    dist = np.sqrt(np.sum((first-second)**2))
    #print("dist:", dist)
    return dist
    
        
    
    
def ideucliDist(A, B, idx):
    first = np.load(A) #把之前存的nparray載進來
    second = np.load(B)
    #print("first:", A)
    idx = int(idx)
    first_idx = first[idx]
    second_idx = second[idx]
    #print(first_idx)
    #print(second_idx)
    dist = np.sqrt(np.sum((first_idx-second_idx)**2))
    #print("dist:", dist)
    return dist

def pic_eucliDist(id1, id2, A):
    first = np.load(A)
    id1 = int(id1)
    id2 = int(id2)
    first_1 = first[id1][1:]
   # print(first_1)
    first_2 = first[id2][1:]
 #   print(first_2)
    dist = np.sqrt(np.sum((first_1-first_2)**2))
    #print("dist:", dist)
    return dist


def eucliDist_no(A, B, id1, id2): #id1和id2是指哪邊到哪邊要用
    first = np.load(A)
    second = np.load(B)
    id1 = int(id1)
    id2 = int(id2)
    first_1 = first[id1:id2+1]
  #  print(first_1)
    second_1 = second[id1:id2+1]
 #   print(second_1)
    dist = np.sqrt(np.sum((first_1-second_1)**2))
    #print("dist:", dist)
    return dist

def sin_dist(A,B):# 直接放np.array
    dist = np.sqrt(np.sum((A-B)**2))
    
#eucliDist("./01brainpic/np/frame000_key.npy", "./01brainpic/np/frame157_key.npy")

# def peak_eucli(A, B):
    
#     for 
#     first = np.load(A)
#     second = np.load(B)
#     dist = np.sqrt(np.sum((first-second)**2))
#     print("dist:", dist)
#     return dist
#pic_eucliDist(19,32,"./me14/pic/me14/np/frame114_key.npy")
#pic_eucliDist(19,32,"./me14/pic/me14/np/frame128_key.npy")
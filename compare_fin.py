# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:27:56 2021

@author: chomi
"""

import os
import random as r
import tensorflow.keras
from PIL import Image
import numpy as np
import cv2 as cv

def find(string):
    path = 'C:/Users/chomi/Desktop/ten/z'
    for i in range(10):
        if(string in os.listdir(path + str(i))):
            return i
def compare(name1, name2):
    im1 = Image.open('C:/Users/chomi/Desktop/all_crop/'+name1)
    im2 = Image.open('C:/Users/chomi/Desktop/all_crop/'+name2)
    size = (800, 800)
    im1 = im1.resize(size, Image.ANTIALIAS)
    im2 = im2.resize(size, Image.ANTIALIAS)
    newim = Image.new('RGB', (1600, 800), (250,250,250))
    newim.paste(im1,(0,0))
    newim.paste(im2,(800,0))
    
    image = newim.resize((224, 224), Image.ANTIALIAS)
    image_array = np.asarray(image)
   
    # cvim = cv.cvtColor(np.array(newim), cv.COLOR_RGB2BGR)
    # cvim = cv.resize(cvim, (800, 400), interpolation = cv.INTER_AREA)
    # cv.imshow('ere', cvim)
    # cv.waitKey(0)
    
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    pred = model.predict(data)
    idx = np.where(pred[0] == np.amax(pred[0]))
    
    if(idx[0][0] == 0):
        print('left')
    elif(idx[0][0] == 1):
        print('right')
    elif(idx[0][0] == 2):
        print('roughly equal')
    return idx[0][0]

def show(n1, n2):
    im1 = Image.open('C:/Users/chomi/Desktop/all_crop/'+n1)
    im2 = Image.open('C:/Users/chomi/Desktop/all_crop/'+n2)
    size = (800, 800)
    im1 = im1.resize(size, Image.ANTIALIAS)
    im2 = im2.resize(size, Image.ANTIALIAS)
    newim = Image.new('RGB', (1600, 800), (250,250,250))
    newim.paste(im1,(0,0))
    newim.paste(im2,(800,0))
    cvim = cv.cvtColor(np.array(newim), cv.COLOR_RGB2BGR)
    cvim = cv.resize(cvim, (800, 400), interpolation = cv.INTER_AREA)
    cv.imshow('ere', cvim)
    key = cv.waitKey(0)
    if(key == ord('1')):
        return 0
    elif(key == ord('2')):
        return 1
    cv.destroyAllWindows()
np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model('compV3.h5', compile = False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

alist = os.listdir('C:/Users/chomi/Desktop/all_crop')

tC = []
tW = []
correct = wrong = sim = 0

# while(correct + wrong < 1000):
#     an = r.randint(0, len(alist)-1)
#     bn = r.randint(0, len(alist)-1)
    
#     n1 = alist[an]
#     n2 = alist[bn]
#     loc1 = find(n1)
#     loc2 = find(n2)
#     if(loc1 == None or loc2 == None):
#         continue
#     truth = loc1 < loc2
#     if(loc1 == loc2):
#         truth = 2
    
#     L = compare(n1, n2)
#     if(truth == 2):
#         sim +=1
#     elif(L == truth and truth != 2):
#         correct+=1
#         tC.append((n1,n2))
#     else:
#         wrong+=1
#         tW.append((n1,n2))
    # key = cv.waitKey(0)
    # if(key == ord('0')):
    #     cv.destroyAllWindows()
    #     break
        

# print('correct:', correct, '   wrong', wrong, '   accuracy:', 100*correct/(correct+wrong), '%')
a = ['A0168-0002.png', 'A0171-0019.png', 'A0165-0020.png', 'A0166-0004.png']
#%%
def insertHelper(st, en, val):
    mid = int((en - st) / 2)
    mid = st + mid
    if(compare(val,  a[mid])):
        return[st, mid]
    else:
        return[mid, en]
        
def insert(list_, val):
    idx = (0, len(list_)-1)
    while(True):
        idx = insertHelper(idx[0], idx[1], val)
        if(idx[1]-idx[0] == 1):
            if(compare(val, list_[idx[0]])):
                idx[1] = idx[0] 
            elif(compare(list_[idx[1]], val)):
                idx[1] += 1
            break
    # list_.insert(idx[1], val)
    print(idx[1] / 10)
    cv.waitKey(0)
alist = os.listdir('C:/Users/chomi/Desktop/all_crop')

for i in range(95):
    pos = r.randint(0, len(alist)-1)
    image_name = alist[pos]
    t = cv.imread('C:/Users/chomi/Desktop/all_crop/'+image_name)
    cv.imshow('jjii', t)
    insert(a, image_name)
    
 

#%%
for i in range(len(tW)):
    loo = tC[i]
    print(loo[0], loo[1])
    print(find(loo[0]), find(loo[1]))
    show(loo[0], loo[1])
cv.destroyAllWindows()

#%%

for i in range(len(a)):
    name = a[i]
    img = cv.imread('C:/Users/chomi/Desktop/all_crop/' + name)
    cv.imshow('wind', img)
    key = cv.waitKey(0)
    if(key == ord('0')): 
        break
cv.destroyAllWindows()
#%%

for i in range(1, len(a)):
    name = a[i]
    j = i-1
    prev = a[j]
    while(j >= 0 and compare(name, prev)):
          a[j+1] = a[j]
          j-= 1
          prev = a[j]
    a[j+1] = name

#%%
while(True):
    pos = r.randint(0, len(alist)-1)
    image_name = alist[pos]
    




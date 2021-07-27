# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 10:50:21 2021

@author: chomi
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.keras
from PIL import Image
import numpy as np
import json

np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('compV3.h5', compile = False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
with open('sorted.txt', 'r') as rp:
    list_ = json.load(rp)
sortedPath = 'sorted/'

def compare(img1, img2):
    img1 = img1.resize((800, 800), Image.ANTIALIAS)
    img2 = img2.resize((800, 800), Image.ANTIALIAS)
    
    newim = Image.new('RGB', (1600, 800), (250,250,250))
    newim.paste(img1,(0,0))
    newim.paste(img2,(800,0))
    
    image = newim.resize((224, 224), Image.ANTIALIAS)
    image_array = np.asarray(image)    
    
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    pred = model.predict(data)
    idx = np.where(pred[0] == np.amax(pred[0]))
    
    return idx[0][0]

def findHelper(st, en, val):
    mid = int((en - st) / 2)
    mid = st + mid
    L = val
    R = Image.open(sortedPath+list_[mid])
    if(compare(L, R)):
        return[st, mid]
    else:
        return[mid, en]
        
def find_position(listt, val):
    idx = (0, len(listt)-1)
    while(True):
        idx = findHelper(idx[0], idx[1], val)
        if(idx[1]-idx[0] == 1):
            temp = Image.open(sortedPath+list_[idx[0]])
            temp2 = Image.open(sortedPath+list_[idx[1]])
            if(compare(val, temp)):
                idx[1] = idx[0] 
            elif(compare(temp2, val)):
                idx[1] += 1
            break
    return(idx[1]/10)

if(__name__ == "__main__"):
    image1 = Image.open('C:/Users/chomi/Desktop/all_crop/A0155-0008.png')
    image2 = Image.open('C:/Users/chomi/Desktop/all_crop/A0166-0004.png')
    result = compare(image1, image2)
    if(result == 0):
        print('image1 is more')
    elif(result == 1):
        print('image2 is more')
    
    g = 'A0179-0003.png'
    it = Image.open('C:/Users/chomi/Desktop/all_crop/'+g)
    ins = find_position(list_, it)
    print('Volume is',ins)



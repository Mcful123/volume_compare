# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 08:16:33 2021

@author: chomi
"""

import os
import random as r
import cv2 as cv 
import json

def find(string):
    path = 'C:/Users/chomi/Desktop/ten/z'
    for i in range(10):
        if(string in os.listdir(path + str(i))):
            return i
    
# ranked = pd.DataFrame()
# done = []
path = 'C:/Users/chomi/Desktop/ten/z'

# alist = os.listdir('C:/Users/chomi/Desktop/all_crop')
# permu = list(it.permutations(alist, 2))

with open('permu.txt', 'r') as rp: #load
    permu = json.load(rp)
    
lnth = len(permu)
while(lnth != 0):
    num = r.randint(0, lnth)    
    nm1 = permu[num][0]
    nm2 = permu[num][1]
    del permu[num]
    loc1 = find(nm1)
    loc2 = find(nm2)
    if(loc1 == None or loc2 == None):
        continue
    
    img1 = cv.imread(path+str(loc1)+'/'+nm1)
    img2 = cv.imread(path+str(loc2)+'/'+nm2)
    
    img1 = cv.resize(img1, (800,800), interpolation = cv.INTER_AREA)
    img2 = cv.resize(img2, (800,800), interpolation = cv.INTER_AREA)
    
    imcom = cv.hconcat([img1, img2])
    imcom = cv.resize(imcom, (224,224), interpolation = cv.INTER_AREA)
    if(loc1 < loc2):
        q = nm1.split('.png')
        newn = q[0]+'_'+nm2
        cv.imwrite('C:/Users/chomi/Desktop/ten/RIGHT/'+newn, imcom)
    elif(loc1 > loc2):
        q = nm2.split('.png')
        newn = q[0]+'_'+nm1
        cv.imwrite('C:/Users/chomi/Desktop/ten/LEFT/'+newn, imcom)
    elif(loc1 == loc2):
        q = nm2.split('.png')
        newn = q[0]+'_'+nm1
        cv.imwrite('C:/Users/chomi/Desktop/ten/EQUAL/'+newn, imcom)
    lnth = len(permu)
    
with open('permu.txt', 'w') as wp: #save
    json.dump(permu, wp)


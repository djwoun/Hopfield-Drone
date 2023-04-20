# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:02:50 2023

@author: djwou
"""


import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import time
f = 12
n = f*f
pattern = 10
bipolarVector = np.random.choice([-1, 1], size=(pattern,n))

inner_matrix = np.zeros((pattern, 100))
      
for i in range (10):
    #print(i)
    for j in range (12):
        for k in range(12):
            if(j==11 or k ==11 or j == 0 or k ==0):
                #print("a")
                bipolarVector[i][j+k*12] = -1
            
            
for i in range(pattern):
    for j in range(1, 11):
        for k in range(1, 11):
            index = (j-1) * 10 + (k-1)
            inner_matrix[i, index] = bipolarVector[i, j*f+k]
    fig = plt.figure()
    plt.axis('off')
    image = (inner_matrix[i].reshape(10,10) + 1) / 2 
    plt.imshow(image, cmap='gray')

#time.sleep(2)
for RUN in range (10):
    fig = plt.figure()
    #print(RUN)
    image = (bipolarVector[RUN].reshape(f, f) + 1) / 2 # Map -1 to 0 and 1 to 1
    #image = (inner_matrix[RUN].reshape(10,10) + 1) / 2 # Map -1 to 0 and 1 to 1
    plt.imshow(image, cmap='gray')
    df =  pd.DataFrame()

    
    plt.axis('off')
    
    plt.savefig("csv2/"+str(RUN), dpi=500)
    #df.reset_index(drop=True, inplace=True)
    #df2.reset_index(drop=True, inplace=True)
    df = pd.DataFrame({str(RUN):  inner_matrix[RUN]})
    #df2 = pd.concat([df2,df], axis=1)
    
#print(df2)

    df.to_csv('csv2/test'+str(RUN)+'.csv', index=False)

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:02:50 2023

@author: djwou
"""


import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

f = 12
n = f*f
pattern = 10
bipolarVector = np.random.choice([-1, 1], size=(pattern,n))

for i in range (10):
    #print(i)
    for j in range (12):
        for k in range(12):
            if(j==11 or k ==11 or j == 0 or k ==0):
                #print("a")
                bipolarVector[i][j+k*12] = -1


#df2 =  pd.DataFrame()
for RUN in range (10):
    fig = plt.figure()
    #print(RUN)
    image = (bipolarVector[RUN].reshape(f, f) + 1) / 2 # Map -1 to 0 and 1 to 1
    
    #plt.show()
    
    df =  pd.DataFrame()
    #plt.set(xlabel=None)
    
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.savefig("csv2/"+str(RUN), dpi=500)
    #df.reset_index(drop=True, inplace=True)
    #df2.reset_index(drop=True, inplace=True)
    df = pd.DataFrame({str(RUN):  bipolarVector[RUN]})
    #df2 = pd.concat([df2,df], axis=1)
    
#print(df2)

    #df.to_csv('csv2/test'+str(RUN)+'.csv', index=False)

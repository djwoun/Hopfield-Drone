# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:02:50 2023

@author: djwou
"""


import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

f = 10
n = f*f
pattern = 10
bipolarVector = np.random.choice([-1, 1], size=(pattern,n))


#df2 =  pd.DataFrame()
for RUN in range (10):
    fig = plt.figure()
    print(RUN)
    image = (bipolarVector[RUN].reshape(f, f) + 1) / 2 # Map -1 to 0 and 1 to 1
    plt.imshow(image, cmap='gray')
    #plt.show()
    df =  pd.DataFrame()
    plt.savefig(str(RUN))
    #df.reset_index(drop=True, inplace=True)
    #df2.reset_index(drop=True, inplace=True)
    df = pd.DataFrame({str(RUN):  bipolarVector[RUN]})
    #df2 = pd.concat([df2,df], axis=1)
    
#print(df2)

    df.to_csv('test'+str(RUN)+'.csv', index=False)

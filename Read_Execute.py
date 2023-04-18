
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from djitellopy import Tello
# Sigma sign function
def sign(x):
    if (x >= 0):
        return 1
    else:
        return -1

# Calculates for the new network with weight
def run(network, weights):
    sizeofNet = len(network)
    new_network = np.zeros(sizeofNet)
    
    for i in range (sizeofNet):
        h = 0
        for j in range (sizeofNet):
            h += weights[i,j] * network[j]
            #print("ASDF" +str(h))
        new_network[i] = sign(h)
    return new_network

bipolarVector = []
for RUN in range (10):
    temp = np.loadtxt("test"+str(RUN)+".csv",
                 delimiter=",", dtype=int)
    temp = np.delete(temp, 0, 0)
    bipolarVector.append(temp)
    #print(bipolarVector)
    
#print(bipolarVector)
df2 =  pd.DataFrame()
for RUN in range (1):
    n = 100
    pattern = 10
    
    fig = plt.figure()


    image = (bipolarVector[0].reshape(10, 10) + 1) / 2 # Map -1 to 0 and 1 to 1
    
    plt.imshow(image, cmap='gray')
    plt.show()
    
    #for p in range(1, pattern+1):
        
    #imprint the weights up to pattern p
    weights = np.zeros((n, n))  
   
    for p2 in range(pattern):
        for i in range(n):
            for j in range(n):
                if i != j:
                    weights[i, j] += bipolarVector[p2][i] * bipolarVector[p2][j] 
        (print(bipolarVector[p2]))
    weights /= n
        
        
    

    tello = Tello()
    
    tello.connect()
    tello.takeoff()
    
    tello.move_left(100)
    tello.rotate_counter_clockwise(90)
    tello.move_forward(100)
    
    tello.land()
        

       
    for i in range(10):
        new_network = run(bipolarVector[i], weights)     
        if np.array_equal(bipolarVector[i], new_network):
            print(i)
    



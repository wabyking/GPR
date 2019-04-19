# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math

import random
for i in [8,14,30]:
    x=np.arange(-1,np.pi*7,0.01)
    y=np.cos(x/math.pow(10000,i/100)) *(2*random.random() +1)
    
    plt.plot(x,y)
#    plt.plot(x,[margin] * len(x),linestyle='dotted')


def plot_dimention(dim,margin):


    x=np.arange(0,np.pi*10,0.01)
    margin = margin + 3
    y=np.sin(x/math.pow(10000,dim/100)) + (margin+0.1)
    
    plt.plot(x,y)

    return margin

#margin=plot_dimention(10,margin)
#margin=plot_dimention(20,margin)
#margin=plot_dimention(50,margin)


## draw embedding


for i in range(0,20):
    y=np.arange(-3,3,0.01)
    x=[i] * len(y)   
    
    plt.plot(x,y,label='line '+str(i),linestyle='dashed',color='gray')
x=np.arange(-1,np.pi*7,0.01)
plt.plot(x,[0] * len(x),linestyle='-',color='black')
y=np.arange(-3.5,3.5)
plt.plot([0] * len(y),y,linestyle='-',color='black')
plt.axis('off')
plt.show()
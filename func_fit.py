# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import data_helper as dh

X = np.array([1,2,3,4,5,6]) 
Y = np.array([9.1,18.3,32,47,69.5,94.8])

def func(params, x): 
    a, b, c = params 
    return a * x * x + b * x + c

def error(params, x, y):
    return func(params, x) - y


def slovePara():
    p0 = [10, 10, 10]

    Para = leastsq(error, p0, args=(X, Y))
    return Para

def solution():
    Para = slovePara()
    a, b, c = Para[0]
    print("a=", a, "b=", b," c=", c)
    print("cost:" + str(Para[1]))
    print("The object curve is: ")
    print("y="+str(round(a, 2))+"x*x+"+str(round(b, 2))+"x+"+str(c))
    
    plt.figure(figsize=(8,6))
    plt.scatter(X, Y, color="green", label="sample data", linewidth=2)
    
    # Draw the fitted line
    x=np.linspace(0,12,100)
    y=a*x*x+b*x+c
    plt.plot(x,y,color="red",label="solution line",linewidth=2)
    plt.legend()
    plt.show()
    
solution()

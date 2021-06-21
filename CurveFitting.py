#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:05:52 2020

@author: skyler
"""
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import numpy as np

#Fitting function
def func(x, a, b):
    
    return a * np.exp(b * x)
    #return a * x + b

#Experimental x and y data points
xData = np.array([1, 2, 3, 4, 5,])
yData = np.array([1, 9, 50, 300, 1500])

#Plot experimental data points
plt.plot(xData, yData, 'bo', label = 'experimental-data')

#Initial guess for the parameters
#initial guess = [1.0, 1.0]

#Perform the curve-fit
popt, pcov = curve_fit(func, xData, yData)
print(popt)

#x values for the fitted function
xFit = np.arrange(0.0, 5.0, 0.01)

#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), 'r', label = 'fit params: a = %5.3f, b = %5.3' % tuple(popt))
    
plt.xlable('x')
plt.ylable('y')
plt.legend()
plt.show()
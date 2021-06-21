#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:59:37 2020

@author: skyler
"""


import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
y = [50000, 17500, 13500, 35000, 525000, 172500]

plt.plot(x, y)
plt.xlabel('Days')
plt.ylabel('Cell Growth/ml')
plt.title('Growth Curve')
plt.show()
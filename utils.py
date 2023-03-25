# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 00:45:39 2023

@author: NickMao
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model, x, y):

    M, N = 500, 500
    x0, x1 = np.meshgrid(np.linspace(x[:, 0].min()*0.95, x[:, 0].max()*1.05, M), np.linspace(x[:, 1].min()*0.95, x[:, 1].max()*1.05, N))
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    z = y_predict.reshape(x0.shape)
    # from matplotlib.colors import ListedColormap
    # custom_cmap = ListedColormap(['#D3D3D3', '#DCDCDC', '#AFEEEE'])
    x_c0 = x[np.where(y==0)]
    x_c1 = x[np.where(y==1)]
    
    plt.figure()
    plt.contourf(x0,x1,z,cmap='jet', alpha = 0.1) 
    plt.pcolormesh(x0, x1, z, cmap='jet', alpha = 0.1)
    plt.scatter(x_c0[:,0], x_c0[:,1], marker="+", c='r', s=50)
    plt.scatter(x_c1[:,0], x_c1[:,1], marker="o", c='b', s=50)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary of Binary Classification")

def sigmoid(x):
    
    return 1/(1+np.e**(-x))

    
    
    
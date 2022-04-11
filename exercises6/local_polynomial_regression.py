# -*- coding: utf-8 -*-
"""
@author: BeatriceCantoni
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class local_Linear:
    'Takes...'
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = len(y)
        self.f = np.zeros(self.n)
        self.weights = np.zeros((self.n, self.n))
    
    def kernel_function(self, distance, h):
        val = distance/h
        return (1/ np.sqrt(2*np.pi)) * np.exp(-0.5* (val**2))
    
    def fit_target(self, target, h): #x star = x[i]
        v = np.ones(self.n)*target
        distance = self.X - v
        k = self.kernel_function(distance, h) /h
        s2 = np.sum(k*((np.ones(self.n)*target-self.X)**2))
        s1 = (np.ones(self.n)*target-self.X)*np.sum(k*(np.ones(self.n)*target-self.X))
        weights = k*(s2-s1)
        weights = weights/np.sum(weights)
        fit = np.sum(weights*self.y)
        return fit, weights
        
    
    def fit(self,h):
        y_fit = []
        #H = np.zeros((self.n, self.n))
        for i in range(self.n):
            sol = self.fit_target(self.X[i], h)
            f = sol[0]
            self.weights[i,:] = sol[1]
            y_fit.append(f)
        self.f = y_fit
        return y_fit
            
    
    def plot(self, h):
        self.f = self.fit(h)
        plt.plot(self.X, self.f, linewidth= 3, color='orange')
        plt.scatter(self.X, self.y, linewidth= 0.1, color = 'blue' )
        plt.title(f'local regression fit, h={h}')
        plt.figure()
        
    def LOOCV(self):
        self.X_training = self.X
        self.y_training = self.y
        self.X_testing = self.X
        self.y_testing = self.y
        H = [0.1, 0.5, 1, 2]
        mse = []
        n = len(self.y)
        for i in range(len(H)):
            plt.subplot(2, 2, i+1)
            y_hat = self.fit(H[i])
            #should correct this
            matrix_mult = self.X.reshape(n,1) @ (1/(self.X.T.reshape(1,n) @ self.X.reshape(n,1))) @ self.X.T.reshape(1,n)
            weights = np.diag(matrix_mult)
            mse.append(np.sum(((self.y - y_hat)/(np.ones(n)-weights))**2))
            
            #plot true vs predicted
            plt.plot(self.X, self.y, label='true')
            plt.plot(self.X, y_hat, label ='prediction')
            plt.title(f'h = {H[i]}')
        plt.legend()
        plt.show()
        
        #compare mse to chose h
        m = mse.index(min(mse))        
        selected_h = H[m]
        print('better chose h=', selected_h)
    
    def get_confidence_intervals(self, h):
        #center of the interval:
        self.f = self.fit(h)       
        #width:
        #hat_s = self.get_hat_sigma_squared(h)
        var_f = np.zeros((self.n))
        #for i in range(self.n):
            #val = np.sum(self.weights[:,i]**2)
            #var_f[i] = var_f[i]*val
        #intervals
        upper = np.zeros(self.n)
        lower = np.zeros(self.n)
        for i in range(self.n):
            var_f[i] = np.sum(self.weights[i, :]**2)
            print(np.sum(self.weights[i, :]))
            upper[i] = self.f[i] + 1.96*np.sqrt(var_f[i])
            lower[i] = self.f[i] - 1.96*np.sqrt(var_f[i])
        return upper, lower

    def plot_intervals(self,h):
        upper, lower = self.get_confidence_intervals(h)
        plt.scatter(self.X, self.y, linewidth=0.1, color='blue')
        plt.plot(self.X, upper, linewidth=2, color='red')
        plt.plot(self.X, self.f, linewidth=2, color='orange')
        plt.plot(self.X, lower, linewidth=2, color='red')
        plt.title('confidence intervals')
           
    def get_hat_sigma_squared(self,h):
        num = np.sum((self.f-self.y)**2)
        H = self.X.reshape(self.n,1) @ (1/(self.X.T.reshape(1,self.n) @ self.X.reshape(self.n,1))) @ self.X.T.reshape(1,self.n)
        s = num/(self.n - np.trace(H) + np.trace(H.T @ H))
        return s
        

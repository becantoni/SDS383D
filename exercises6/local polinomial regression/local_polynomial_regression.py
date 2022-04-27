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
        self.residuals = np.zeros(self.n)
    
    def kernel_function(self, distance, h):
        val = distance/h
        return (1/ np.sqrt(2*np.pi)) * np.exp(-0.5* (val**2))
    
    def fit_target(self, target, h): 
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
        for i in range(self.n):
            sol = self.fit_target(self.X[i], h)
            f = sol[0]
            self.weights[i,:] = sol[1]
            self.residuals[i] = self.y[i] - f
            self.f[i] = f
            
    
    def plot(self, h):
        self.fit(h)
        plt.plot(self.X, self.f, linewidth= 3, color='orange')
        plt.scatter(self.X, self.y, s=10, color = 'blue' )
        plt.title(f'local regression fit, h={h}')
        plt.figure()
        
    def LOOCV(self):
        self.X_training = self.X
        self.y_training = self.y
        self.X_testing = self.X
        self.y_testing = self.y
        H = np.linspace(6, 8, 10)
        mse = []
        n = len(self.y)
        for i in range(len(H)):
            self.fit(H[i])  
            mse.append(np.sum((self.residuals/(np.ones(n)-np.diag(self.weights)))**2)) 
            
        
        #compare mse to chose h
        m = mse.index(min(mse))        
        selected_h = H[m]
        print('better chose h=', selected_h)
        
    def estimate_sigma_squared(self): #unbiased correction       
        num = np.sum(self.residuals**2) - self.f.T @ (
            np.identity(self.n) - self.weights).T @ (
            np.identity(self.n) - self.weights) @ self.f
        den = self.n - 2*np.trace(self.weights) + np.trace(self.weights.T @ self.weights)
        sigma_s = num/den
        print(sigma_s)
        return sigma_s
    
    def get_confidence_intervals(self, h):
        #center of the interval:
        self.fit(h)     #fit and update  self.f
        #width:
        var_f = np.zeros((self.n))
        upper = np.zeros(self.n)
        lower = np.zeros(self.n)
        ss = self.estimate_sigma_squared()
        for i in range(self.n):                        
            var_f[i] = np.sum(self.weights[i, :]**2)
            upper[i] = self.f[i] + 1.96*np.sqrt(var_f[i]*ss)
            lower[i] = self.f[i] - 1.96*np.sqrt(var_f[i]*ss)
        print(var_f[0])
        return upper, lower

    def plot_intervals(self,h):
        upper, lower = self.get_confidence_intervals(h)
        plt.scatter(self.X, self.y, s=10, color='blue')
        plt.plot(self.X, upper, linewidth=2, color='red')
        plt.plot(self.X, self.f, linewidth=2, color='orange')
        plt.plot(self.X, lower, linewidth=2, color='red')
        plt.title(f'confidence intervals, h= {h}')
           
    def get_hat_sigma_squared(self,h):
        num = np.sum((self.f-self.y)**2)
        H = self.X.reshape(self.n,1) @ (
            1/(self.X.T.reshape(1,self.n) @ self.X.reshape(self.n,1))) @ self.X.T.reshape(1,self.n)
        s = num/(self.n - np.trace(H) + np.trace(H.T @ H))
        return s
        
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 11:07:46 2022

@author: BeatriceCantoni
"""

import numpy as np
import matplotlib.pyplot as plt

class kernel_regression:
    'Takes...'
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def kernel_function(self, distance, h):
        val = distance/h
        return (1/ np.sqrt(2*np.pi)) * np.exp(-0.5* (val**2))
        
    def weight_function(self, x, x_star, h):
        v = np.ones(len(x))*x_star
        distance = x - v
        result = self.kernel_function(distance, h) /h
        return result
    
    def plot_k_reg(self, X, y, h):
        x_star = np.linspace(min(X),max(X), 500)
        y_fit = []
        for i in range(len(x_star)):
            weights = self.weight_function(X, x_star[i], h)
            weights = weights/np.sum(weights)
            y_fit.append(np.sum(weights*y))
        plt.plot(x_star, y_fit, color='blue')
        plt.title(f'h = {h}')
        plt.figure()
    
    def plot_compared_h(self):
        hs = [0.2, 0.5, 1, 5]
        for i in range(len(hs)):
            self.plot_k_reg(self.X, self.y, hs[i])
    
    def predict(self, h):
        y_predict =[]
        for d in range(len(self.X_testing)):
            weights = self.weight_function(self.X_training, 
                                 self.X_testing[d],
                                 h)
            weights = weights/np.sum(weights)
            y_predict.append(np.sum(weights*self.y_training))
        return y_predict, weights
    
    def split_dataset(self):        
        index = [item for item in range(0, len(self.y))]
        np.random.shuffle(index)
        cut = int(len(self.y)) * 4/5
        training_index = index[: int(cut)]
        training_index = np.sort(training_index) #order in acent order for plot purpose
        testing_index = index[int(cut) :]
        testing_index = np.sort(testing_index)
        self.X_training = self.X[training_index]
        self.y_training = self.y[training_index]
        self.X_testing = self.X[testing_index]
        self.y_testing = self.y[testing_index]
            
            
    def h_train(self):
        H = [0.1, 0.2, 0.5, 1, 2]
        mse = []
        y_hat = []
        for i in range(len(H)):
            plt.subplot(2, 2, i+1)
            self.split_dataset()
            y_hat.append(self.predict(H[i])[0])
            mse.append(np.mean((y_hat[i]-self.y_testing)**2))
            #plot true vs predicted
            plt.plot(self.X_testing, self.y_testing, label='true')
            plt.plot(self.X_testing, y_hat[i], label ='prediction')
            plt.title(f'h = {H[i]}')
        plt.show()
        #compare mse to chose h
        m = mse.index(min(mse))        
        selected_h = H[m]
        print('better chose h=', selected_h)
    
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
            y_hat = self.predict(H[i])[0]
            #option 1
            print(self.X.shape, self.X.T.shape)
            matrix_mult = self.X.reshape(n,1) @ (self.X.T.reshape(1,n) @ self.X.reshape(n,1))**(-1) @ self.X.T.reshape(1,n)
            weights = np.diag(matrix_mult)
            #option 2
            #weights = self.predict(H[i])[1]
            mse.append(np.sum(((self.y - y_hat)/(np.ones(n)-weights))**2))
            
            #plot true vs predicted
            plt.plot(self.X, self.y, label='true')
            plt.plot(self.X, y_hat, label ='prediction')
            plt.title(f'h = {H[i]}')
            #plt.legend()
        plt.show()
        
        #compare mse to chose h
        m = mse.index(min(mse))        
        selected_h = H[m]
        print('better chose h=', selected_h)
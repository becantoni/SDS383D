# -*- coding: utf-8 -*-
"""
@author: BeatriceCantoni
"""

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.optimize import minimize


class GP_model:
    ''''Takes: numpy vector of x points
        and vecor of hyperparameters
        and fit GP model (posterior/predictive/marginal..)'''
    def __init__(self, X, y):
        self.X = X
        self.N = len(y)
        self.y = y
        self.get_distance_matrix()
        self.get_s2()
        
    def get_distance_matrix(self):
        out = distance_matrix(self.X,self.X)
        self.D = out
        
    def get_s2(self): 
        self.s2 = .7
    
    def cov_exp(self, par, D):
        tau_sq_1, tau_sq_2, b = par
        C = tau_sq_1 * np.exp(-1 / 2 * (D / b) ** 2) 
        C = C + tau_sq_2 * np.eye(C.shape[0], C.shape[1])
        return C

    def cov_52(self, par, D):
        tau_sq_1, tau_sq_2, b = par
        C = tau_sq_1 * (1 + np.sqrt(5) * D / b + 5 * D ** 2 / (3 * b ** 2)) 
        C = C * np.exp(-np.sqrt(5) * D / b) + tau_sq_2 * np.eye(C.shape[0], C.shape[1])
        return C        
    
    def find_optimal_parameters(self):
        o = minimize(lambda parameters: -self.log_marginal(parameters), 
                 x0=np.array([1, 1]), 
                 bounds=((0, None),(0, None)), method='Powell')    
        return o
    
    '''functions for fitting parameters'''
    
    def log_marginal(self, p):    
        par = [p[0], 10**(-6), p[1]]
        cov_fun = self.cov_exp(par, self.D) #
        cov_norm = self.s2*np.eye(self.N) + cov_fun
        dist = multivariate_normal(cov=cov_norm, allow_singular=True)
        marginal = dist.logpdf(self.y)
        return marginal
    
    def parameters_grid(self, tau_1_range, b_range):
        #prepare grid
        w = len(tau_1_range)
        tt, bb = np.meshgrid(tau_1_range, b_range)
        all_tau = tt.flatten()
        all_b = bb.flatten()#
        m_grid = np.array([self.log_marginal([x, y]) for (x,y) in zip(all_tau, all_b)])
        
        #get values of parameters that maximise the log marginal
        optim = np.argwhere(m_grid == np.max(m_grid))
        print(f'highest when tau_1^2 = {all_tau[optim]}, b = {all_b[optim]}')
        
        #plot 2d grid
        plt.figure(figsize=(8, 6))
        plt.contour(tt, bb, m_grid.reshape(w,w), 125, cmap='RdGy')
        plt.scatter(all_tau[optim], all_b[optim], color='black', marker='+', s=100)
        plt.title('log marginal on 2d grid of possible parameters')
        plt.xlabel('tau_1^2')
        plt.ylabel('b')
        
    
    
    
    
    '''functions for predictive'''  
    
    def predictive_parameters(self, par, X_star, cov_type):       
        #get necessary elements
        D12 = distance_matrix(X_star, self.X) 
        C12 = cov_type(par, D12)
        D21 = D12.T
        C21 = cov_type(par, D21)
        D22 = distance_matrix(X_star, X_star)
        C22 = cov_type(par, D22)
        C11 = cov_type(par, self.D) + self.s2*np.eye(self.N)
        inv = np.linalg.pinv(C11)
        
        #get mean and cov of the predictive:
        H = C12 @ inv
        mean = H @ self.y 
        cov = C22 - (C12 @ inv @ C21)
        
        return mean, cov
        
    

    '''functions to get posteriors'''
    def posterior_mean(self, C): 
        self.H = np.linalg.inv(C + self.s2*np.identity(self.N))@ C
        mean = self.H @self.y
        return mean
    
    def posterior_variance(self, C): 
        var = np.linalg.inv(((1/self.s2)*np.identity(self.N) + np.linalg.inv(C)))
        return var


    '''functions for posterior plots'''   
    #posterior plot when x is 2d
    def posterior_contour(self, cov_type): 
        g1 = np.linspace(min(self.X[:,0]), max(self.X[:,0]), 100)
        g2 = np.linspace(min(self.X[:,1]), max(self.X[:,1]), 100)
        g1_grid, g2_grid = np.meshgrid(g1, g2)
        X_gen = np.stack([g1_grid.flatten(), g2_grid.flatten()]).T
        
        p_optim = self.find_optimal_parameters().x
        print('parameters used = ', p_optim)
        parameters = [p_optim[0], 10**(-6), p_optim[1]]
        pred = self.predictive_parameters(parameters, X_gen, cov_type)
        f_pred = pred[0]
        sq_pred = np.sqrt(np.diag(pred[1]))
        w = len(g1)
        
        #plot 2d grid
        #posterior mean
        plt.figure(figsize=(8, 6))
        plt.contourf(g1, g2, f_pred.reshape(w,w), 125, cmap='RdGy')
        plt.colorbar()
        plt.scatter(self.X[:,0], self.X[:,1], color='black', s=10)
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('posterior mean over full grid')
        #3dgrid
        plt.figure(figsize=(8, 6))
        ax = plt.axes(projection='3d')
        ax.contour3D(g1, g2, f_pred.reshape(w,w), 150, cmap='binary')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('3d visulization of posterior mean over full grid')
        #posterior standard deviation
        plt.figure(figsize=(8, 6))
        plt.contourf(g1, g2, sq_pred.reshape(w,w), 125, cmap='RdGy')
        plt.colorbar()
        plt.scatter(self.X[:,0], self.X[:,1], color='black', s=10)
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('posterior sd over full grid')
    
    #posterior plot when x is 1d
    def get_intervals(self, parameters, cov_type): 
        C = cov_type(parameters, self.D)
        mean = self.posterior_mean(C)
        S = self.posterior_variance(C)
        variances = np.diag(S)
        width = [np.sqrt(x)*1.96 for x in variances]
        lower = mean - width
        upper = mean + width
        
        plt.scatter(self.X, self.y, s=10, color='blue')
        plt.plot(self.X, mean, color='orange')
        plt.plot(self.X, lower, color='red')
        plt.plot(self.X, upper, color='red')
        plt.title('Posterior credible intervals')
        plt.show()
        
        





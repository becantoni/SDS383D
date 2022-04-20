# -*- coding: utf-8 -*-
"""
@author: BeatriceCantoni
"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import squareform
import pandas as pd
import matplotlib.pyplot as plt

class gaussian_process_simulator:
    ''''Takes: numpy vector of x points
        and hyperparameters
        Returns: simulated value of the process at each point'''
    def __init__(self, points, parameters):
        self.X = points
        self.N = len(points)
        self.parameters = parameters
        self.get_distance_matrix()
        
    def get_distance_matrix(self):
        # mesh this array so that you will have all combinations
        m, n = np.meshgrid(self.X, self.X)
        # get the distance via the norm
        out = abs(m-n)
        self.D = out
        
    def mean_function(self):
        return np.zeros(self.N)    
    
    def cov_squared_exponential(self, par):
        tau_s_1, tau_s_2, b = par[0], par[1], par[2]
        first = tau_s_1 * np.exp(-0.5*(self.D**2) /b**2)
        matrix = first + np.identity(self.N)*tau_s_2
        return matrix
    
    def simulate_gp_sq_exp(self, par):
        m = self.mean_function()
        v = self.cov_squared_exponential(par)
        simul = multivariate_normal.rvs(m,v)
        return simul        
    
    def cov_52(self, par):
        tau_s_1, tau_s_2, b = par[0], par[1], par[2]
        #print(tau_s_1, tau_s_2, b)
        first = tau_s_1 * np.exp(-np.sqrt(5)*self.D /b) #manca un pezzo qui?
        first = first*(1+ np.sqrt(5)*self.D/b) + 5*(self.D**2)/(3*(b**2))
        cov_matrix = first + np.identity(self.N)*tau_s_2
        return cov_matrix
    
    def simulate_gp_52(self, par):
        m = self.mean_function()
        v = self.cov_52(par)
        simul = multivariate_normal.rvs(m,v)
        return simul

    '''plot functions'''
    def plot_compare(self, c_type, par_index, par_to_try):     
        params = self.parameters.copy()
        for t in range(len(par_to_try)):
            plt.subplot(1, 3, t+1)
            params[par_index] = par_to_try[t]
            trial = gaussian_process_simulator(self.X, params)
            for t in range(4):
                y = c_type(params) 
                plt.plot(self.X, y)
            plt.title(f'$ tau_{1}^{2}$ = {params[0]}, $ tau_{2}^{2}$ = {params[1]}, $b$ = {params[2]}')
        plt.show()
        
          
        

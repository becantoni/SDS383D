# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 10:18:20 2022
@author: Beatrice Cantoni
"""
#%%import packages
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import gamma
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class LinearModel_Bayes:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.p = self.X.shape[1]
        self.n = self.X.shape[0]
    
    def initialise_parameters(self):
        self.K = np.identity(self.p)*.00001
        self.m = np.zeros(self.p)
        self.d = 1
        self.eta = 1
        self.h = 1
        
    #get updated parameters
    def _get_K_star(self, lam, K):
        K_star = self.X.T@lam@self.X+K
        return K_star

    def _get_m_star(self, lam, K, m):
        K_star = self._get_K_star(lam, K)
        m = m.reshape(6,1)
        m_star = np.linalg.inv(K_star)@(self.X.T@lam@self.y+K@m)
        return m_star
    
    def _get_d_star(self):
        return self.d+self.n

    def _get_eta_star(self, eta, lam, m, K):
        m_star = self._get_m_star(lam, K, m)
        K_star = self._get_K_star(lam, K)
        eta_star = eta + self.y.T@lam@y + m.T@K@m - m_star.T@K_star@m_star
        return eta_star

    def fit_homoskedastic(self, fit_intercept=True):
        if fit_intercept:
            column_ones = np.ones(self.X.shape[0])[:, None]
            self.X = np.hstack((column_ones, self.X))
            self.p = self.p +1
        
        self.initialise_parameters()
        self.lam = np.identity(self.n)
        
        self.m_star = self._get_m_star(self.lam, self.K, self.m)
        self.K_star = self._get_K_star(self.lam, self.K)
        self.d_star = self._get_d_star(self.d, self.n)
        self.eta_star = self._get_eta_star(self.eta, self.lam, self.m, self.K)
        #get scale parameter
        K_to_use = np.linalg.inv(linalg.sqrtm(self.K_star))
        self.scale_star = (self.eta_star/self.d_star) * K_to_use
        
    
    def _update_lambda(self):
        alpha = (self.h + 1) / 2
        beta = 1 / 2*(self.h + self.w  * ((self.y).reshape(self.n,) - self.X @ self.beta))
        samples = gamma.rvs(alpha, beta)
        self.lam = np.diag(samples)

    def _update_beta(self, m_star, precision_matrix):
        self.beta = multivariate_normal(mean=m_star.reshape(6,),
                                        cov=linalg.inv(precision_matrix)).rvs()

    def _update_w(self, d_star, eta_star):
        self.w = gamma.rvs(d_star / 2, 2 / eta_star)
    
    def fit_heteroskedastic(self, n_iter, fit_intercept = True):
        if fit_intercept:
            column_ones = np.ones(self.X.shape[0])[:, None]
            self.X = np.hstack((column_ones, self.X))
            self.p = self.p +1
            
        #initialize parameters
        self.initialise_parameters()
        self.lam = np.diag(np.ones(self.n))
        self.beta = np.ones(self.p)*0.5
        self.w = 1
        
        #housekeeping setup
        self.traces = {'beta_trace': np.zeros([n_iter, self.X.shape[1]]),
            'Lambda_trace': np.zeros([n_iter, self.X.shape[0]]),
            'omega_trace': np.zeros(n_iter)
            }
        
        it = 1
        d_star = self._get_d_star() #doesn't need to be updated anymore
        #update parameters iteratively        
        while it <= n_iter:
            #print(self.lam.shape)
            eta_star = self._get_eta_star(self.eta, self.lam, self.m, self.K)
            K_star = self._get_K_star(self.lam, self.K)            
            m_star = self._get_m_star(self.lam, self.K, self.m)
            precision_matrix = self.w * K_star
            try: 
                self._update_lambda()
                self._update_beta(m_star, precision_matrix)
                self._update_w(d_star, eta_star)
            except ValueError:
                pass
            it = it +1
            print(it)
            
            #store values
            #print(self.lam.shape)
            self.traces['Lambda_trace'][it, :] = np.diag(self.lam)
            self.traces['beta_trace'][it, :] = self.beta
            self.traces['omega_trace'][it] = self.w
                

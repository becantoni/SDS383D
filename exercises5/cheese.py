# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 22:06:33 2022

@author: BeatriceCantoni
"""

import numpy as np
from scipy.stats import multivariate_normal, gamma, norm, f
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


class Hierarchical:
    '''Takes list of design matrices (one per store)
    and list of vectors (one per store)'''
    # initialise the object

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.P = len(y)
        self.n_par = 4
        self.N = 5555  # make it better

    def initialise_parameters(self):
        self.beta = np.zeros((self.n_par, self.P))
        self.gam = np.zeros(self.P)
        self.sigma_squared = 1
        self.tau_squared = 1

    def _update_sigma_squared(self, gam, b, tau_squared):
        alpha = (self.P + self.N) / 2
        b_sum = 0
        for p in range(0, self.P):
            val = self.X[p] @ b[:, p]
           #print(y[p] - val)
            b_sum = b_sum + (self.y[p] - val).T @ (
                self.y[p] - val) + (b[:, p] - gam).T @ (b[:, p] - gam) / tau_squared
        beta = 1 / 2 * b_sum
        self.sigma_squared = 1 / gamma.rvs(a=alpha, scale=1 / beta, size=1)

    def _update_tau_squared(self, gam, b, sigma_squared):
        alpha = self.P + 0.5
        b_sum = 1
        for p in range(0, self.P):
            b_sum = b_sum + (b[:, p] - gam).T @ (b[:, p] - gam) / sigma_squared
        beta = 0.5 * (b_sum)
        self.tau_squared = 1 / gamma.rvs(a=alpha, scale=1 / beta, size=1)

    def _update_gam(self, sigma_squared, tau_squared, b):
        var = np.identity(self.n_par) * (sigma_squared * tau_squared / self.P)
        mean = var @ (np.sum(np.array(b), axis=1)) / \
            (sigma_squared * tau_squared)
        self.gam = multivariate_normal.rvs(mean, var)
        
    def _update_beta(self, gam, sigma_squared, tau_squared):
        for p in range(0, self.P):
            V = np.linalg.inv(self.X[p].T @ self.X[p] / sigma_squared +
                              np.identity(self.n_par) / (sigma_squared * tau_squared))
            m = V @ (self.X[p].T @ self.y[p] / sigma_squared +
                     gam/(tau_squared * sigma_squared))
            self.beta[:, p] = multivariate_normal.rvs(mean=m, cov=V)

    def _update_traces(self, it):
        self.traces['betas'][it, :, :] = self.beta
        self.traces['gamma'][it] = self.gam
        self.traces['sigma_squared'][it] = self.sigma_squared
        self.traces['tau_squared'][it] = self.tau_squared

    def fit_GibbsSampler(self, n_iter, burn):
        # initialize parameters:
        self.initialise_parameters()
        # housekeeping setup
        self.n_iter = n_iter
        self.burn = burn
        self.traces = {'sigma_squared': np.zeros(self.n_iter),
                       'tau_squared': np.zeros(self.n_iter),
                       'gamma': np.zeros((self.n_iter, self.n_par)),
                       'betas': np.zeros((self.n_iter, self.n_par, self.P)),
                       }
        # do gibbs steps:
        for it in tqdm(range(self.n_iter)):
            self._update_gam(self.sigma_squared, self.tau_squared, self.beta)
            self._update_sigma_squared(self.gam, self.beta, self.tau_squared)
            self._update_tau_squared(self.gam, self.beta, self.sigma_squared)
            self._update_beta(self.gam, self.sigma_squared, self.tau_squared)
            self._update_traces(it)
        # remove burnin:
        for trace in self.traces.keys():
            self.traces[trace] = self.traces[trace][self.burn:]
            
    '''Define plot functions'''    
    def plot_beta_histograms_1(self):        
        for b in range(self.n_par): 
            traces = self.traces['betas']
            for group in range(self.P):
                plt.hist(traces[:, b, group], density=True, alpha=.5, bins=50)
            plt.title(f'beta {b} posteriors, all stores')
            plt.figure()
    
    def plot_beta_histograms_2(self):
        for group in range(self.P):
            traces = self.traces['betas']
            for b in range(self.n_par):
                plt.hist(traces[:, b, group], density=True, alpha=.5, bins = 50)
            plt.title('beta posteriors')
            plt.figure()
    
    def plot_beta_store(self, index):
        traces = self.traces['betas']
        for b in range(self.n_par):
            plt.hist(traces[:, b, index], density=True, alpha=.5, bins = 50)
        plt.title('beta posteriors')
        plt.figure()
        
    def plot_gamma_histogram(self):
        traces = self.traces['gamma']
        for b in range(self.n_par): #change that 2
            plt.hist(traces[:, b], density= True, alpha=.5, bins=50)
        plt.title('gamma posterior')
        plt.figure()
        

    def plot_other_histograms(self, variable):
        trace = self.traces[variable]
        plt.hist(trace, density=True, alpha=.5, bins=50)
        plt.title(f'{variable} posterior')
        plt.figure()

    def plot_all_posteriors(self):
        self.plot_beta_histograms_1()
        self.plot_gamma_histogram()
        keys = list(self.traces.keys())
        keys.remove('betas')
        keys.remove('gamma')
        for trace in keys:
            self.plot_other_histograms(trace)
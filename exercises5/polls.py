# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 20:21:40 2022

@author: BeatriceCantoni
"""

import numpy as np
from scipy.stats import multivariate_normal, gamma, norm, bernoulli, truncnorm
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
        self._n_par = self.X[0].shape[1]
        self._n_groups = len(self.X)
                

    def initialise_parameters(self):
        self.m = np.ones(self._n_par)
        self.tau_squared = 1
        self.z = [np.random.choice([0, 1], size=len(x)) for x in self.y]
        self.gam = np.ones((self._n_par, self._n_groups))
        self._lower_bounds, self._upper_bounds = self._get_trunc_normal_params()
        

    def _get_trunc_normal_params(self):
        lower_bounds = []
        upper_bounds = []
        for group in range(self._n_groups):
            lower_bound = [-np.inf if x == 0 else 0 for x in self.y[group]]
            upper_bound = [np.inf if x == 1 else 0 for x in self.y[group]]
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        return lower_bounds, upper_bounds
        
    def _update_gamma(self):
        for group in range(self._n_groups):
            cov = np.linalg.inv(self.tau_squared * np.eye(self._n_par) + self.X[group].T @ self.X[group])
            mean = cov @ (1 / self.tau_squared * self.m + self.X[group].T @ self.z[group])
            self.gam[:, group] = multivariate_normal.rvs(mean=mean, cov=cov)            
            
    def _update_tau_squared(self):
        alpha = (self._n_groups + 1) / 2
        beta = 0.5 * (((self.gam - self.m[:, None]) ** 2).sum() + 1)
        self.tau_squared = 1 / gamma.rvs(a=alpha, scale=1 / beta)

    def _update_m(self):
        cov = self.tau_squared / self._n_groups * np.eye(self._n_par)
        mean = cov / self.tau_squared @ self.gam.sum(axis=1)
        self.m = multivariate_normal.rvs(mean=mean, cov=cov)

    def _update_z(self):
        self.z = []
        for group in range(self._n_groups):
            self.z.append(truncnorm.rvs(self._lower_bounds[group], self._upper_bounds[group], loc=0, scale=1))
            
    def _update_traces(self, it):
        self.traces['gammas'][it, :, :] = self.gam
        self.traces['m'][it, :] = self.m
        self.traces['tau_squared'][it] = self.tau_squared

    
    
    def fit_GibbsSampler(self, n_iter, burn):
        # initialize parameters:
        self.initialise_parameters()
        # housekeeping setup
        self.n_iter = n_iter
        self.burn = burn
        self.traces = {'gammas': np.zeros((n_iter, self._n_par, self._n_groups)),
                       'tau_squared': np.zeros(n_iter),
                       'm': np.zeros((n_iter, self._n_par))}

        # do gibbs steps:
        for it in tqdm(range(self.n_iter)):
            self._update_m()
            self._update_tau_squared()
            self._update_gamma()
            self._update_z()
            self._update_traces(it)
        # remove burnin:
        keys = list(self.traces.keys())
        for key in keys:
            self.traces[key] = self.traces[key][self.burn:]

        
        '''Define plot functions'''    
    def _plot_gamma_histograms(self):
        for j in range(self._n_par):
            for i in range(self._n_groups):
                plt.hist(self.traces['gammas'][:, j, i], density=True, alpha=.5, bins=50)
            plt.title(f'coefficient {j + 1}', fontsize=16)
            plt.figure()

    def _plot_tau_histograms(self):
        plt.hist(self.traces['tau_squared'], density=True, alpha=.5, bins=50)
        plt.title('tau_squared', fontsize=16)
        plt.figure()

    def _plot_m_histogram(self):
        for i in range(self._n_par):
            plt.hist(self.traces['m'][:, i], density=True, alpha=.5, bins=50)
        plt.title('m', fontsize=16)
        plt.figure()

    def plot_all_histograms(self):
        self._plot_m_histogram()
        self._plot_tau_histograms()
        self._plot_gamma_histograms()

    def plot_all_traces(self):
        keys = list(self.traces.keys())
        plt.figure(figsize=(8, 9))
        for i, key in enumerate(keys):
            plt.subplot(len(keys), 1, i + 1)
            plt.title(key, fontsize=16)
            plt.xticks([])
            plt.yticks([])
            if key == 'gammas':
                gammas_traces = self.traces['gammas']
                gammas_traces = gammas_traces.reshape(gammas_traces.shape[0],
                                                    gammas_traces.shape[1] * gammas_traces.shape[2])
                plt.plot(gammas_traces, color='dodgerblue', alpha=.5)
            else:
                plt.plot(self.traces[key], color='dodgerblue', alpha=.7)
        
        
        
        

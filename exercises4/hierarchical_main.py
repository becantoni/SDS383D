# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:36:48 2022

@author: BeatriceCantoni
"""
import numpy as np
from scipy.stats import multivariate_normal, gamma, norm, f
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

class Hierarchical:
    '''Takes a dataframe and fits a Hirerchical model'''
    '''Data should be given as already grouped using the groupby function
    and with only one column of values'''
    #initialise the object
    def __init__(self, df):
        self.df = df
        self.df.columns = ['group', 'values']
        self.P = self.df.groupby('group').ngroups
        self.n_i = self.df.groupby('group').size().to_numpy()
        self.mean_per_group = self.df.groupby('group').mean().to_numpy().flatten()
        self.N = self.n_i.sum()
    
    def _update_tau_squared(self, dist_mu):
        alpha = (self.P + 1) / 2
        beta = 1 / 2 * ((self.theta - dist_mu).T @ np.linalg.inv(self.sigma_squared * np.identity(self.P)) @ (
                self.theta - dist_mu) + 1)
        self.tau_squared = 1 / gamma.rvs(a=alpha, scale=1 / beta, size=1)

    def _update_sigma_squared(self, dist_mu):
        alpha = (self.N + self.P) / 2
        beta_first_term = (self.theta - dist_mu).T @ np.linalg.inv(self.tau_squared * np.identity(self.P)) @ (
                self.theta - dist_mu)
        beta_second_term = ((self.theta[self.df['group'].values - 1] - self.df['values']) ** 2).sum()
        beta = 0.5 * (beta_first_term + beta_second_term)
        self.sigma_squared = 1 / gamma.rvs(a=alpha, scale=1 / beta, size=1)

    def _update_mu(self):
        mean = self.theta.mean()
        var = self.sigma_squared * self.tau_squared / self.P
        self.mu = norm.rvs(loc=mean, scale=np.sqrt(var))

    def _update_theta(self, dist_mu):
        mean = (self.mean_per_group * self.tau_squared * self.n_i + dist_mu) / (self.tau_squared * self.n_i + 1)
        cov_matrix = self.sigma_squared * self.tau_squared / (self.tau_squared * self.n_i + 1) * np.identity(self.P)
        self.theta = multivariate_normal.rvs(mean=mean, cov=cov_matrix)

    def _update_traces(self, it):
        self.traces['theta'][it, :] = self.theta
        self.traces['ki'][it, :] = self.ki
        self.traces['mu'][it] = self.mu
        self.traces['sigma_squared'][it] = self.sigma_squared
        self.traces['tau_squared'][it] = self.tau_squared
        
    def get_ki(self):
        ki = (1/self.n_i)/(self.tau_squared*np.ones(self.P) + 1/self.n_i)
        self.ki = ki
        
        
            
    def fit_GibbsSampler(self, n_iter, burn):
        #initialize parameters:
        self.theta = self.mean_per_group
        self.ki = np.zeros(self.P)
        self.mu = 0
        self.sigma_squared = 1
        self.tau_squared = 1
        # housekeeping setup
        self.n_iter = n_iter
        self.burn = burn
        self.traces = {'sigma_squared': np.zeros(self.n_iter),
                       'tau_squared': np.zeros(self.n_iter),
                       'mu': np.zeros(self.n_iter),
                       'theta': np.zeros((self.n_iter, self.P)),
                       'ki': np.zeros((self.n_iter, self.P))}
        #do gibbs steps:
        for it in tqdm(range(self.n_iter)):
            self._update_mu()
            self._update_sigma_squared(self.mu)
            self._update_tau_squared(self.mu)
            self._update_theta(self.mu)
            self.get_ki()
            self._update_traces(it)
        #remove burnin:
        for trace in self.traces.keys():
            self.traces[trace] = self.traces[trace][self.burn:]
    
    def plot_theta_histograms(self):
        traces = self.traces['theta']
        for group in range(self.P):
            plt.hist(traces[:, group], density=True, alpha=.5, bins=50)
        plt.title('theta posteriors')
        plt.figure()
        
    def get_estimate(self):
        self.theta_post = []
        traces = self.traces['theta']
        for group in range(self.P):
            self.theta_post.append(np.mean(traces[:, group]))
        trace = self.traces['tau_squared']
        self.tau_squared_post = np.mean(trace)
        trace = self.traces['sigma_squared']
        self.sigma_squared_post = np.mean(trace)
        self.mu_post = np.mean(self.traces['sigma_squared'])
        
    
    def plot_other_histograms(self, variable):
        trace = self.traces[variable]
        plt.hist(trace, density=True, alpha=.5, bins=50)
        plt.title(f'{variable} posterior')
        plt.figure()

    def plot_all_posteriors(self):
        self.plot_theta_histograms()
        keys = list(self.traces.keys())
        keys.remove('theta')
        keys.remove('ki')
        for trace in keys:
            self.plot_other_histograms(trace)
    
    def plot_means(self):
        means_posterior = np.mean(self.traces['theta'], axis=0)
        means_data = self.mean_per_group
        plt.plot([i+1 for i in range(self.P)], 
                 means_posterior, linewidth =1.0, label='posterior')
        plt.plot([i+1 for i in range(self.P)], 
                 means_data, linewidth =1.0, label='data')
        plt.title('sample vs posterior theta: shrinkage effect?')
        plt.legend()
        plt.figure()
        
    def shrinkage(self):
        ki_means = np.mean(self.traces['ki'], axis=0)
        plt.scatter(self.n_i, ki_means)
        plt.xlabel('group size')
        plt.ylabel('shrinkage coefficient')
        plt.title('Shrinkage effect')

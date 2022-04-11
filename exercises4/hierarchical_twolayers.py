# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:19:08 2022

@author: BeatriceCantoni
"""

import numpy as np
from scipy.stats import multivariate_normal, gamma, norm, t
from scipy import stats
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


class Hierarchical:
    '''Takes a dataframe and fits a Hirerchical model'''
    '''Dataframe should have value ordered as:
        date-value-subject-group(treatment)'''
    #initialise the object
    def __init__(self, df):
        self.df = df
        self.df.columns = ['_', 'values', 'subject', 'group']
        self.P = self.df.groupby('subject').ngroups
        self.n_i = self.df.groupby('subject').size().to_numpy()
        auxiliary_df = self.df[['subject', 'values']]
        self.mean_per_group = auxiliary_df.groupby('subject').mean().to_numpy().flatten()
        self.N = self.n_i.sum()
        #initialise design matrix
        first_column = np.ones(self.P)
        second_column = np.array([np.zeros(10), 
                                  np.ones(10)])
        second_column = second_column.reshape(self.P,)
        self.X = np.stack([first_column, second_column], axis=1)
        self.n_treat_groups = 2 
    
    def _update_tau_squared(self, gam):
        dist_mu = self.X @ gam
        alpha = (self.P + 1) / 2
        beta = 1 / 2 * ((self.theta - dist_mu).T @ np.linalg.inv(self.sigma_squared * np.identity(self.P)) @ (
                self.theta - dist_mu) + 1)
        self.tau_squared = 1 / gamma.rvs(a=alpha, scale=1 / beta, size=1)

    def _update_sigma_squared(self, gam):
        dist_mu = self.X @ gam
        alpha = (self.N + self.P) / 2
        beta_first_term = (self.theta - dist_mu).T @ np.linalg.inv(self.tau_squared * np.identity(self.P)) @ (
                self.theta - dist_mu)
        beta_second_term = ((self.theta[self.df['subject'].values - 1] - self.df['values']) ** 2).sum()
        beta = 0.5 * (beta_first_term + beta_second_term)
        self.sigma_squared = 1 / gamma.rvs(a=alpha, scale=1 / beta, size=1)

    def _update_gam(self):
        mean = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.theta
        var = self.sigma_squared * self.tau_squared * np.linalg.inv(self.X.T @ self.X)
        self.gam = multivariate_normal.rvs(mean, var)

    def _update_theta(self, gam):
        dist_mu = self.X @ gam
        mean = (self.mean_per_group * self.tau_squared * self.n_i + dist_mu) / (self.tau_squared * self.n_i + 1)
        cov_matrix = self.sigma_squared * self.tau_squared / (self.tau_squared * self.n_i + 1) * np.identity(self.P)
        self.theta = multivariate_normal.rvs(mean=mean, cov=cov_matrix)

    def _update_traces(self, it):
        self.traces['theta'][it, :] = self.theta
        self.traces['gamma'][it] = self.gam
        self.traces['sigma_squared'][it] = self.sigma_squared
        self.traces['tau_squared'][it] = self.tau_squared
        
        
    def fit_GibbsSampler(self, n_iter, burn):
        #initialize parameters:
        self.theta = self.mean_per_group
        self.gam = np.zeros(self.n_treat_groups) 
        self.sigma_squared = 1
        self.tau_squared = 1
        # housekeeping setup
        self.n_iter = n_iter
        self.burn = burn
        self.traces = {'sigma_squared': np.zeros(self.n_iter),
                       'tau_squared': np.zeros(self.n_iter),
                       'gamma': np.zeros((self.n_iter, self.n_treat_groups)), 
                       'theta': np.zeros((self.n_iter, self.P)),
                       }
        #do gibbs steps:
        for it in tqdm(range(self.n_iter)):
            self._update_gam()
            self._update_sigma_squared(self.gam)
            self._update_tau_squared(self.gam)
            self._update_theta(self.gam)
            self._update_traces(it)
        #remove burnin:
        for trace in self.traces.keys():
            self.traces[trace] = self.traces[trace][self.burn:]
    
    '''Define plot functions'''    
    def plot_theta_histogram(self):
        traces = self.traces['theta']
        for group in range(self.P):
            plt.hist(traces[:, group], density=True, alpha=.5, bins=50)
        plt.title('theta posteriors')
        plt.figure()
        
    def plot_gamma_histogram(self):
        traces = self.traces['gamma']
        for treatment in range(2): #change that 2
            plt.hist(traces[:, treatment], density= True, alpha=.5, bins=50)
        plt.title('beta and mu posterior')
        plt.figure()
        

    def plot_other_histograms(self, variable):
        trace = self.traces[variable]
        plt.hist(trace, density=True, alpha=.5, bins=50)
        plt.title(f'{variable} posterior')
        plt.figure()

    def plot_all_posteriors(self):
        self.plot_theta_histogram()
        self.plot_gamma_histogram()
        keys = list(self.traces.keys())
        keys.remove('theta')
        keys.remove('gamma')
        for trace in keys:
            self.plot_other_histograms(trace)
    
    def plot_means(self):
        means_posterior = np.mean(self.traces['theta'], axis=0)
        means_data = self.mean_per_group
        plt.plot([i+1 for i in range(self.P)], 
                 means_posterior, linewidth =2.0, label='posterior')
        plt.plot([i+1 for i in range(self.P)], 
                 means_data, linewidth =1.0, label='data')
        plt.title('sample vs posterior theta: shrinkage effect?')
        plt.legend()
        plt.figure()

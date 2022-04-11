# -*- coding: utf-8 -*-
'''
@author: Beatrice Cantoni
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

#%%
class GLM:
    '''Class for GLM that can be built with different link functions'''
    def __init__(self, X, y):
        self.X = X
        self.y = y
        # Define epsilon because log(0) is not defined
        self.eps = 1e-7
        
    def standardize(self):
        '''standardize/scale X'''
        means = self.X.mean(axis=0)
        stds = self.X.std(axis=0)
        return (self.X - means) / stds
        
        
    def sigmoid(self, z): #function specific for this glm
        '''
        Takes a numpy array 
        applies sigmoid function to every element
        returns numpy array
        '''            
        sig_z = (1/(1+np.exp(-z)))        
        return sig_z
    

    def log_likelihood(self, beta):
        '''
        Takes: values of parameters, numpy array
        Returns: Log-likelihood, scalar value
        '''
        # get expected value according to beta
        z = np.dot(self.X, beta)       
        mu = self.sigmoid(z)
        #Modify 0/1 values in mu so that log is not undefined
        #important when it goes in the denominator
        mu = np.maximum(np.full(mu.shape, self.eps), np.minimum(np.full(mu.shape, 1-self.eps), mu))
            
        likelihood = sum(self.y*np.log(mu)+(1-self.y)*np.log(1-mu))            
        return likelihood
    
    def gradient(self, beta):
        z = np.dot(self.X, beta)
        mu = self.sigmoid(z)
        gradient = np.sum((self.y-mu)*self.X.T, axis=1)
        return gradient
    
    def hessian(self, beta):
        z = np.dot(self.X, beta)       
        mu = self.sigmoid(z)
        #get a diagonal matrix with results in the diagonal
        diag = mu*(1-mu)
        W = np.diag(diag)
        hessian = - self.X.T @ W @ self.X
        return hessian
    
    def gradient_ascent(self):
        '''Trains logistic regression model using gradient ascent'''
        self.X = self.standardize() #standardize before adding intercept
        self.X = np.c_[np.ones(len(self.X[:,1])), self.X] #add intercept
        
        #initialize
        self.likelihoods  = []
        num_features = self.X.shape[1]
        iterations = 1
        step_size_relative = 1
        learning_rate = 0.001
        
        # Initialize beta with appropriate shape        
        self.beta = np.zeros(num_features)
        
        # Perform gradient ascent
        while step_size_relative >= 10**(-6):
            
            gradient = self.gradient(self.beta)

            # Update the beta
            self.beta = self.beta + learning_rate*gradient
                        
            # get log likelihood
            likelihood = self.log_likelihood(self.beta)
            self.likelihoods.append(likelihood)
            
            #update number of iterations:
            iterations = iterations + 1
            if iterations >= 5:
                step_size_relative = np.abs(
                    self.likelihoods[-1] - self.likelihoods[-2])/np.abs(self.likelihoods[-2])
                
        print('Converged in', iterations, 'iterations')
    
    
    def newton_rapson(self):
        self.X = self.standardize() #standardize before adding intercept
        self.X = np.c_[np.ones(len(self.X[:,1])), self.X] #add intercept
        
        #initialise
        self.likelihoods  = []
        num_features = self.X.shape[1]
        iterations = 1
        step_size_relative = 1
        
        
        # Initialize beta with appropriate shape        
        self.beta = np.zeros(num_features)
        
        # Perform gradient ascent
        while step_size_relative >= 10**(-6):
            #get the gradient
            gradient = self.gradient(self.beta)
            #get the hessian
            H = self.hessian(self.beta)
            # Update the beta
            #use solve (LU decomposition) instead of inverting
            self.beta = self.beta - np.linalg.solve(H, gradient) 
            # get log likelihood
            likelihood = self.log_likelihood(self.beta)
            self.likelihoods.append(likelihood)
            
            #update number of iterations:
            iterations = iterations + 1
            if iterations >= 5:
                step_size_relative = np.abs(
                    self.likelihoods[-1] - self.likelihoods[-2])/np.abs(self.likelihoods[-2])
                
        print('Converged in:', iterations, 'iterations')

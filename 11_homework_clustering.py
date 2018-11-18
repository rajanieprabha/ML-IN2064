
# coding: utf-8

# # Programming assignment 11: Gaussian Mixture Model

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

from scipy.stats import multivariate_normal


# ## Your task

# In this homework sheet we will implement Expectation-Maximization algorithm for learning & inference in a Gaussian mixture model.
# 
# We will use the [dataset](http://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat) containing information about eruptions of a geyser called "Old Faithful". The dataset in suitable format can be downloaded from Piazza.
# 
# As usual, your task is to fill out the missing code, run the notebook, convert it to PDF and attach it you your HW solution.

# ## Generate and visualize the data

# In[2]:

X = np.loadtxt('faithful.txt')
plt.figure(figsize=[6, 6])
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('Eruptions (minutes)')
plt.ylabel('Waiting time (minutes)')
plt.show()


# ## Task 1: Normalize the data
# 
# Notice, how the values on two axes are on very different scales. This might cause problems for our clustering algorithm. 
# 
# Normalize the data, such that it lies in the range $[0, 1]$ along each dimension (each column of X).

# In[3]:

def normalize_data(X):
    """Normalize data such that it lies in range [0, 1] along every dimension.
    
    Parameters
    ----------
    X : np.array, shape [N, D]
        Data matrix, each row represents a sample.
        
    Returns
    -------
    X_norm : np.array, shape [N, D]
        Normalized data matrix. 
    """
    #print(X)
    # (X-mu)sigma
    
    #X_norm = ((X - np.mean(X,axis = 0))/ np.std(X, axis =0))
    X_norm = (X - np.min(X,axis =0))/(np.max(X,axis = 0)- np.min(X,axis =0))
    #X_norm[:,1] = ((X[:,1] - np.mean(X[:,1],axis = 1))/ np.std(X[:,1], axis =1))
    #X_norm = np.absolute(X_norm)
    #print(X_norm)
    
    
    return X_norm


# In[4]:

plt.figure(figsize=[6, 6])
X_norm = normalize_data(X)
plt.scatter(X_norm[:, 0], X_norm[:, 1]);


# ## Task 2: Compute the log-likelihood of GMM
# 
# Here and in some other places, you might want to use the function `multivariate_normal.pdf` from the `scipy.stats` package.

# In[22]:

def gmm_log_likelihood(X, means, covs, mixing_coefs):
    """Compute the log-likelihood of the data under current parameters setting.
    
    Parameters
    ----------
    X : np.array, shape [N, D]
        Data matrix with samples as rows.
    means : np.array, shape [K, D]
        Means of the GMM (\mu in lecture notes).
    covs : np.array, shape [K, D, D]
        Covariance matrices of the GMM (\Sigma in lecture notes).
    mixing_coefs : np.array, shape [K]
        Mixing proportions of the GMM (\pi in lecture notes).
        
    Returns
    -------
    log_likelihood : float
        log p(X | \mu, \Sigma, \pi) - Log-likelihood of the data under the given GMM.
    """
    log_likelihood = None
    K = means.shape[0]
    
    for i in range(K):
        gaus = (multivariate_normal.pdf(X, means[i], covs[i]))
        #x = np.multiply(mixing_coefs[i],gaus)
    llike = np.log(gaus)
    
    #print(gaus.shape)
           
        
    
    log_likelihood = np.sum(llike)
    #print(log_likelihood)
    
    
    
    return log_likelihood


# ## Task 3: E step

# In[18]:

def e_step(X, means, covs, mixing_coefs):
    """Perform the E step.
    
    Compute the responsibilities.
    
    Parameters
    ----------
    X : np.array, shape [N, D]
        Data matrix with samples as rows.
    means : np.array, shape [K, D]
        Means of the GMM (\mu in lecture notes).
    covs : np.array, shape [K, D, D]
        Covariance matrices of the GMM (\Sigma in lecuture notes).
    mixing_coefs : np.array, shape [K]
        Mixing proportions of the GMM (\pi in lecture notes).
    
    Returns
    -------
    responsibilities : np.array, shape [N, K]
        Cluster responsibilities for the given data.
    """
    
    K = means.shape[0]
    N = X.shape[0]
    D = X.shape[1]
    responsibilities = np.zeros((N,K))
    gaus = np.zeros((N,D))
    
    #check = np.zeros(K)
    #Sum = 0
    
    for i in range(K):
        for j in range(N):
            responsibilities[j][i] = mixing_coefs[i] * (multivariate_normal.pdf(X[j], means[i], covs[i]))
            #x = mixing_coefs[j]* gaus)
            #Sum = Sum + x
            #responsibilities[i][j] = x/np.sum(x)
        
    #norm = np.sum(x)     
    #print(x.shape)
    #print(responsibilities)
    responsibilities/= responsibilities.sum(0)
    
    
    print(responsibilities.shape)
    
    return responsibilities


# ## Task 4: M step

# In[8]:

def m_step(X, responsibilities):
    """Update the parameters \theta of the GMM to maximize E[log p(X, Z | \theta)].
    
    Parameters
    ----------
    X : np.array, shape [N, D]
        Data matrix with samples as rows.
    responsibilities : np.array, shape [N, K]
        Cluster responsibilities for the given data.
    
    Returns
    -------
    means : np.array, shape [K, D]
        Means of the GMM (\mu in lecture notes).
    covs : np.array, shape [K, D, D]
        Covariance matrices of the GMM (\Sigma in lecuture notes).
    mixing_coefs : np.array, shape [K]
        Mixing proportions of the GMM (\pi in lecture notes).
    
    """
   
    N = X.shape[0]
    K = responsibilities.shape[1]
    D = X.shape[1]
    covs= np.zeros((K,D,D))
    means = np.zeros((K,D))
    mixing_coefs =np.zeros(K)
    Xm = np.zeros((N,D))
    #sqr = np.zeros((D,D))
    
    
    #print(responsibilities.shape)
    
    # Nk = sum of k responsibilities
    Nk = np.sum(responsibilities,axis=0)
    #print(Nk)
    
    
    for k in range(K):
        for n in range(N):
            means[k] += (responsibilities[n][k] * X[n])
            
        means[k] = means[k]/Nk[k]   
        #means[k] = (np.sum(np.dot(responsibilities[:,k],X)))/(Nk[k])
    #print(means.shape)
            
    #means = np.sum(means,)
    #print(means)
    for k in range(K):
        for n in range(N):
            Xm[n] = (X[n] - means[k])
            sqr = np.dot(Xm.T,Xm)
            #print(sqr.shape)
           
            covs[k] += (responsibilities[n][k]*sqr)
            
        covs[k] = covs[k]/Nk[k]
            #covs[k] = covs[k]/Nk[k]
        #print(mid.shape)
        #covs[k] = (Xm * mid)/Nk[k
    for k in range(K):
        
        mixing_coefs[k] = Nk[k]/N
        
    #print(means.shape)
    print(covs.shape)
    
    return means, covs, mixing_coefs


# ## Visualize the result (nothing to do here)

# In[9]:

def plot_gmm_2d(X, responsibilities, means, covs, mixing_coefs):
    """Visualize a mixture of 2 bivariate Gaussians.
    
    This is badly written code. Please don't write code like this.
    """
    plt.figure(figsize=[6, 6])
    palette = np.array(sns.color_palette('colorblind', n_colors=3))[[0, 2]]
    colors = responsibilities.dot(palette)
    # Plot the samples colored according to p(z|x)
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5)
    # Plot locations of the means
    for ix, m in enumerate(means):
        plt.scatter(m[0], m[1], s=300, marker='X', c=palette[ix],
                    edgecolors='k', linewidths=1,)
    # Plot contours of the Gaussian
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(x, y)
    for k in range(len(mixing_coefs)):
        zz = mlab.bivariate_normal(xx, yy, np.sqrt(covs[k][0, 0]),
                                   np.sqrt(covs[k][1, 1]), 
                                   means[k][0], means[k][1], covs[k][0, 1])
        plt.contour(xx, yy, zz, 2, colors='k')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


# ## Run the EM algorithm

# In[23]:

X_norm = normalize_data(X)
max_iters = 20

# Initialize the parameters
means = np.array([[0.2, 0.6], [0.8, 0.4]])
covs = np.array([0.5 * np.eye(2), 0.5 * np.eye(2)])
mixing_coefs = np.array([0.5, 0.5])

old_log_likelihood = gmm_log_likelihood(X_norm, means, covs, mixing_coefs)

responsibilities = e_step(X_norm, means, covs, mixing_coefs)
print('At initialization: log-likelihood = {0}'
      .format(old_log_likelihood))
plot_gmm_2d(X_norm, responsibilities, means, covs, mixing_coefs)

# Perform the EM iteration
for i in range(max_iters):
    responsibilities = e_step(X_norm, means, covs, mixing_coefs)
    means, covs, mixing_coefs = m_step(X_norm, responsibilities)
    new_log_likelihood = gmm_log_likelihood(X_norm, means, covs, mixing_coefs)
    # Report & visualize the optimization progress
    print('Iteration {0}: log-likelihood = {1:.2f}, improvement = {2:.2f}'
          .format(i, new_log_likelihood, new_log_likelihood - old_log_likelihood))
    old_log_likelihood = new_log_likelihood
    plot_gmm_2d(X_norm, responsibilities, means, covs, mixing_coefs)


# In[ ]:




import numpy as np
from numpy import shape
import scipy as sc
import pandas as pd
from ABC_algorithm import ABC

import statsmodels as sm
from sklearn.linear_model import LinearRegression

def EpanechnikovKernel(t,delta,c=1):
    if t<=delta:
        return c*(1/delta)*(1-(t/delta)**2)
    else:
        return 0

def PostProcess(abc_df, Summary, data, q=0.5, weighted=True):
    """
    Given a dataframe output to ABC, return dataframe with accepted thetas transformed in column "theta_star"
    """
    # take accpted values output from ABC
    df_accepted = abc_df[abc_df['accept'] == 1]
    accepted_count = len(df_accepted.index)
    if accepted_count < 5:
    	print accepted_count, "is number of accepted thetas"

    if accepted_count < 2:
    	print "Post processing failed; too few accepted values."
    	return 0

    df_accepted.statistics_diff = np.subtract(df_accepted.statistics,Summary(data)) # here a list of vectors
    df_accepted.statistics_diff_abs = np.power(np.sum(np.square(df_accepted.statistics_diff)), 0.5) # we want a vector of distances
    
    #df_accepted.statistics_diff_abs.hist(bins=100)
    quantile = df_accepted.statistics_diff_abs.quantile(q) # shouled be ok
    #plt.show()

    #create column with kernal transform
    df_accepted.kernel = df_accepted.statistics_diff_abs.apply(lambda x: EpanechnikovKernel(x, delta=quantile)) #diff_abs

    #print df_accepted.kernel[df_accepted.kernel > 0.0]

    mod = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    X = np.array(df_accepted.statistics_diff) #list of vector
    X.shape = (shape(X)[0],1)                 #the number of cols is equal to the lenght of the data
    y = np.array(df_accepted.theta)
    y.shape = (shape(y)[0],1)
    weights = np.array(df_accepted.kernel)
    
    if weighted:
        res = mod.fit(X, y, sample_weight=weights)
    else: 
        res = mod.fit(X, y)
    #alpha = res.intercept_[0]
    beta = res.coef_[0][0]  #dimension equal to the size of the data
    beta_stats_diff = beta*df_accepted.statistics_diff # multiplication of: vector * matrix
    beta_stats_diff_array = np.array(beta_stats_diff)  #change the size
    beta_stats_diff_array.shape = (shape(beta_stats_diff_array)[0],1) #change the size
    theta_star = np.subtract(y, beta_stats_diff_array)
    theta_star = np.reshape(theta_star, theta_star.shape[0])
    df_accepted['theta_star'] = theta_star
    
    return df_accepted
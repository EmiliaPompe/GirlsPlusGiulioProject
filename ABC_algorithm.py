import numpy as np
import scipy as sc
import pandas as pd

def ABC(priorSampler, likelihoodSimulator, summaryStatistics, epsilon, data, n): # epsilon is the number
    # data is a numpy.array (format), each element is one observation 
    # priorSampler and likelihoodSimulator return numpy.arrays, each element is a random variable
    # summaryStatistics returns a 1-dim array
    
    # priorSampler - a function taking one argument: n - the desired length of the sample, it returns an np.array
    # likelihoodSimulator - a function taking two argument: the desired number of observations and the current parameter, it returns an np.array 
    # summaryStatistics - a function taking one argument - data, returns a 1-dim array
    # epsilon - currently a number, not percentage,
    # data - an array
    # n - number of simulations (not the number of accepted samples)
    
    prior_sample = priorSampler(n)
    stat = summaryStatistics(data)
    theta_generated = []
    accepted = []
    output_list = []
    
    for i in range(n):
        # data is currently an array of shape (data_len,)
        simulated_data = likelihoodSimulator(np.shape(data)[0], prior_sample[i])
        
        temporary_stat = summaryStatistics(simulated_data)
        # in the line below we are comparing sum of squares of the elements of temporary_stat - stat
        if np.sum(np.square(np.subtract(temporary_stat, stat))) < epsilon*epsilon: # check here!
            accept = 1
        else: accept = 0
            
        output_dict = {'accept': accept, 'z':simulated_data, 'theta': prior_sample[i]} # added theta
        # seems more reasonable to add the theta at the end of function ...
        output_list.append(output_dict)
    
    df = pd.DataFrame(output_list)
    return df

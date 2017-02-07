import numpy as np
import scipy as sc
import pandas as pd
import timeit
import seaborn as sns
#sns.set_style("whitegrid")
#sns.set_context("talk")
#rc('axes', labelsize=20, titlesize=20)

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import scipy.stats as ss

import timeit

def ABC_sample(priorSampler, likelihoodSimulator, summaryStatistics, epsilon, data, n): 
    # epsilon is the tolerance value
    # data is a numpy.array (format), each element is one observation 
    # priorSampler and likelihoodSimulator return numpy.arrays, each element is a random variable
    # summaryStatistics returns a 1-dim array
    
    # priorSampler - a function taking one argument: n - the desired length of the sample, it returns an np.array
    # likelihoodSimulator - a function taking two argument: the desired number of observations and the current parameter, it returns an np.array 
    # summaryStatistics - a function taking one argument - data, returns a 1-dim array
    # epsilon - currently a number, not percentage,
    # data - an array
    # n - number of accepted samples (NOT number of iterations)
    
    #OUTPUT:
    #It returns a list where the first element is the dataframe (as ABC function)
    #and the second element is the number of iterations needed to create the sample of required size n
    
    stat = summaryStatistics(data)
    theta_generated = []
    accepted = []
    output_list = []
    i = 0
    niter = 0
    iteration_time = []
    
    while True:
        if (niter % 1000 == 0):
            start_time = timeit.default_timer()
        niter = niter+1
        # simulate prior
        simulated_prior = priorSampler(1)
        # data is currently an array of shape (data_len,)
        simulated_data = likelihoodSimulator(np.shape(data)[0], simulated_prior[0])
        
        temporary_stat = summaryStatistics(simulated_data)
        # in the line below we are comparing sum of squares of the elements of temporary_stat - stat
        if np.sum(np.square(np.subtract(temporary_stat, stat))) < epsilon*epsilon: # check here!
            accept = 1
            output_dict = {'accept': accept, 'z':simulated_data, 'theta': simulated_prior[0]} # added theta
            # seems more reasonable to add the theta at the end of function ...
            output_list.append(output_dict)
            i = i+1
        else: accept = 0         
        
        if (niter % 1000 == 0):
            iteration_time.append(timeit.default_timer() - start_time)
        
        if i==n:
            break
    
    df = pd.DataFrame(output_list)
    return (df, niter, iteration_time)


# In[4]:

def weighted_variance(values, weights):
    #Return the weighted variance of values (a np.array)
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    variance = variance*len(values)/(len(values)-1) #Unbias estimator
    return variance


# In[53]:

def ABC_PMC(priorFunction, priorSampler, likelihoodSimulator, summaryStatistics, epsilon_array, data, n): 
    # ABC_PMC allows the user to set a decreasing sequence of tolerance levels. It returns the sample obtained with the last tolerance level.
    
    # epsilon_array is a numpy.array, it should be a decreasing sequence of tolerance values
    # data is a numpy.array (format), each element is one observation 
    # priorSampler and likelihoodSimulator return numpy.arrays, each element is a random variable
    # summaryStatistics returns a 1-dim array
    
    # priorFunction - a function taking one argument and returns the value of the pdf of the prior in that argument
    # priorSampler - a function taking one argument: n - the desired length of the sample, it returns an np.array
    # likelihoodSimulator - a function taking two argument: the desired number of observations and the current parameter, it returns an np.array 
    # summaryStatistics - a function taking one argument - data, returns a 1-dim array
    # epsilon - currently a number, not percentage,
    # data - an array
    # n - number of accepted samples (NOT number of iterations! WATCH OUT)
    
    #OUTPUT:
    #It returns a list where the first element is a list containing the parameters and pseudo-data
    #obtained with the last tolerance leve.
    #The second element is the number of iterations needed to create the sample of required size n
    
    iteration_time = []
    
    #Run basic ABC using the first tolerance level
    temp = ABC_sample(priorSampler, likelihoodSimulator, summaryStatistics, epsilon_array[0], data , n)
    df = temp[0]
    niter = temp[1] #Number of iterations required for the initial step
    weight_old = np.ones(n)*(1/n) #assign "basic" weights to the sampled parameters
    theta_old = df.theta
    sigma_squared = 2*weighted_variance(df.theta,weight_old) #Compute a weighted empirical variance of theta. Used as variance in the Gaussian kernel
    
    stat = summaryStatistics(data) #compute statistics of original data
    output_list = []
    
    for t in range(1,len(epsilon_array)):
        i = 0
        theta_accepted = []
        weight = []
        sigma = math.sqrt(sigma_squared)
        while True:
            if (niter % 1000 == 0):
                start_time = timeit.default_timer()
            niter = niter+1
            theta_star = np.random.choice(theta_old, size = 1, p=weight_old) #get one of the previous theta obtained at random (weighted)
            simulated_prior = np.random.normal(loc = theta_star, scale = sigma, size = 1) #perturbate the choice
            simulated_data = likelihoodSimulator(np.shape(data)[0], simulated_prior[0]) #simulate data
            temporary_stat = summaryStatistics(simulated_data) #get statistics of simulated data
            if np.sum(np.square(np.subtract(temporary_stat, stat))) < epsilon_array[t]*epsilon_array[t]:
                #Accept!
                theta_accepted.append(simulated_prior[0])
                if t==(len(epsilon_array)-1): #last tolerance level, prepare output
                    output_dict = {'z':simulated_data, 'theta': simulated_prior[0]} 
                    output_list.append(output_dict)
                #Compute weight
                #den = 0
                #for j in range(0,n):
                #    aux = (1/sigma)*(simulated_prior[0]-theta_old[j])
                #    den = den + weight_old[j]*(1/sigma)*(1/math.sqrt(2*math.pi))*exp(-(aux)**2/2)
                #weight.append(priorFunction(simulated_prior[0])/den)

                #Faster way to compute weight
                phi = np.true_divide(np.exp(-np.true_divide(np.power(np.true_divide(np.subtract(np.ones(n)*simulated_prior[0],theta_old),sigma),2),2)),math.sqrt(2*math.pi))
                den_new = np.sum(np.true_divide(np.multiply(weight_old,phi),sigma))
                weight.append(priorFunction(simulated_prior[0])/den_new)     
                #End compute weight
                
                i = i+1
                
            if (niter % 1000 == 0 and 'start_time' in vars()):
                iteration_time.append(timeit.default_timer() - start_time)
            if i==n:
                break
        
        weight = weight/np.sum(weight) #normalize weight so that the sum is 1
        sigma_squared = 2*weighted_variance(theta_accepted,weight) #compute sigma given the new weights
        weight_old = weight #save weight for next step
        theta_old = theta_accepted
        
    df = pd.DataFrame(output_list)
    return(df, niter, iteration_time)


# In[6]:


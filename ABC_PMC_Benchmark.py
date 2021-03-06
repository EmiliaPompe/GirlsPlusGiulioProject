
# coding: utf-8

# In[1]:

#get_ipython().magic('pylab inline')
import numpy as np
import scipy as sc
import pandas as pd

#import seaborn as sns
#sns.set_style("whitegrid")
#sns.set_context("talk")
#rc('axes', labelsize=20, titlesize=20)

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import scipy.stats as ss

import timeit
import pickle

from ABC_PMC import ABC_sample, ABC_PMC


# In[2]:

######
# set up for the normal ABC example
######

prior_mean = -4.0
prior_sd = 3
likelihood_sd = 1
original_data_size = 100

def NormalPriorFunction(x):
    return ss.norm.pdf(x=x,loc=prior_mean, scale=prior_sd)

def NormalPriorSampler(n):
    return np.random.normal(loc=prior_mean, scale=prior_sd, size=n)

def NormalLiklihoodSimulator(n, param):
    #unknown mean
    return np.random.normal(loc=param, scale=likelihood_sd, size=n)
    
def NormalSummary(data):
    return np.mean(data, axis=0)

data = np.random.normal(loc=0,scale=likelihood_sd,size=original_data_size)


# In[3]:

k = 5
requested_sample_size = np.linspace(start=50,stop=1000,num=20) #n=50,100,...,1000
tolerance_seq = np.linspace(start=1, stop= 0.01, num=4) #T = 4


#PMC benchmark for different values of T but fixed n
sample_size_seq = [100,250,500,1000]

PMC_benchmark2 = []
for sample_size in sample_size_seq:
    print("PMC 2 - ",float(sample_size_seq.index(sample_size))/len(sample_size_seq)*100,"%")
    for T in range(2,21):
        tolerance_seq = np.linspace(start=1, stop= 0.01, num=T)
        run_time_PMC = 0
        niter_PMC = 0
        times_for_iteration_PMC = 0
        for rep in range(k):
            #Run ABC
            start_time = timeit.default_timer()
            PMC_run = ABC_PMC(NormalPriorFunction,NormalPriorSampler, NormalLiklihoodSimulator, NormalSummary, tolerance_seq, data , sample_size)
            run_time_PMC += timeit.default_timer() - start_time
            niter_PMC += PMC_run[1]
            times_for_iteration_PMC += np.mean(PMC_run[2])
        run_time_PMC /= k
        niter_PMC /= k
        times_for_iteration_PMC /= k
        output_dict = {'T':T, 'sample_size': sample_size, 'run_time': run_time_PMC, 'niter':niter_PMC, 'times_1000_iter': times_for_iteration_PMC}
        PMC_benchmark2.append(output_dict)

file_name = "data/Correct_PMC_benchmark2_k_"+str(k)+"_tol_"+str(tolerance_seq[-1])+"_num_"+str(len(sample_size_seq))+"_T_2_10.p"
pickle.dump(PMC_benchmark2, open( file_name, "wb" ) ) #save result in a file
#print(pd.DataFrame(PMC_benchmark2))
print("PMC 2 - 100%")


# In[ ]:

print("DONE! :D")


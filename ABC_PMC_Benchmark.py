
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

k = 1
requested_sample_size = np.linspace(start=50,stop=250,num=2) #n=50,100,...,1000
tolerance_seq = np.linspace(start=1, stop= 0.01, num=4) #T = 4


# In[13]:

ABC_benchmark = []

for sample_size in requested_sample_size:
    print("ABC - ",float(len(ABC_benchmark))/len(requested_sample_size)*100,"%")
    run_time_ABC = 0
    niter_ABC = 0
    times_for_iteration_ABC = 0
    for rep in range(k):       
        #Run ABC
        start_time = timeit.default_timer()
        ABC_run = ABC_sample(NormalPriorSampler, NormalLiklihoodSimulator, NormalSummary, tolerance_seq[-1], data , sample_size)
        run_time_ABC += timeit.default_timer() - start_time
        niter_ABC += ABC_run[1]
        times_for_iteration_ABC += np.mean(ABC_run[2])
    run_time_ABC /= k
    niter_ABC /= k
    times_for_iteration_ABC /= k
    output_dict = {'sample_size': sample_size, 'run_time': run_time_ABC, 'niter':niter_ABC, 'times_1000_iter': times_for_iteration_ABC}
    ABC_benchmark.append(output_dict)

file_name = "data/ABC_benchmark_k_"+str(k)+"_tol_"+str(tolerance_seq[-1])+"_num_"+str(len(requested_sample_size))+".p"
pickle.dump(ABC_benchmark, open( file_name, "wb" ) ) #save result in a file
#You can load using:
#test = pickle.load( open( "data/ABC_benchmark_k_2_tol_001.p", "rb" ) )

#print(pd.DataFrame(ABC_benchmark))
print("ABC - 100%")



# In[14]:

PMC_benchmark = []

for sample_size in requested_sample_size:
    print("PMC 1 - ",float(len(PMC_benchmark))/len(requested_sample_size)*100,"%")
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
    output_dict = {'sample_size': sample_size, 'run_time': run_time_PMC, 'niter':niter_PMC, 'times_1000_iter': times_for_iteration_PMC}
    PMC_benchmark.append(output_dict)
    
file_name = "data/PMC_benchmark_k_"+str(k)+"_tol_"+str(tolerance_seq[-1])+"_num_"+str(len(requested_sample_size))+"_T_"+str(len(tolerance_seq))+".p"
pickle.dump(PMC_benchmark, open( file_name, "wb" ) ) #save result in a file
#You can load using:
#test = pickle.load( open( "data/ABC_benchmark_k_2_tol_001.p", "rb" ) )

#print(pd.DataFrame(PMC_benchmark))
print("PMC - 100%")


# In[15]:

#PMC benchmark for different values of T but fixed n
sample_size_seq = [100,250,500,1000]

PMC_benchmark2 = []
for sample_size in sample_size_seq:
    print("PMC 2 - ",float(sample_size_seq.index(sample_size))/len(sample_size_seq)*100,"%")
    for T in range(2,11):
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

file_name = "data/PMC_benchmark2_k_"+str(k)+"_tol_"+str(tolerance_seq[-1])+"_num_"+str(len(sample_size_seq))+"_T_2_10.p"
pickle.dump(PMC_benchmark, open( file_name, "wb" ) ) #save result in a file
#print(pd.DataFrame(PMC_benchmark2))
print("PMC 2 - 100%")


# In[ ]:

print("DONE! :D")


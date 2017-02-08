
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

import pickle
import gc 
import timeit
from ABC_algorithm import ABC


# In[2]:


# In[3]:

######
# set up for the g and k distribution
######

def SimulateGK(n, param):  #B>0, K>-1/2 #param is a vector of A, B, g, k
    A, B, g, k = param[0], param[1], param[2], param[3]
    u_values = np.random.uniform(low=0.0, high=1.0, size=n)
    x_values = np.zeros(n)
    for i in range(0,n):
        x_values[i] = A + B*(1+0.8*(1-np.exp(-g*sc.stats.norm.ppf(u_values[i], 0, 1) )) /(1 + np.exp(-g*sc.stats.norm.ppf(u_values[i], 0, 1) ))) *np.power((1+ np.power(sc.stats.norm.ppf(u_values[i], 0, 1),2)),k)*(sc.stats.norm.ppf(u_values[i], 0, 1))    
    return x_values

def GKPriorSampler(n):
    l = []
    for i in range(n):
        l.append(np.random.uniform(low=0.0, high=10.0, size=4)) # we assume uniform [1,10] prior for all 4 parameters
    return l

def GKLiklihoodSimulator(n, param):
    #unknown mean
    return SimulateGK(n, param)
    
def GKSummary(data):
    return np.sort(data)


data_gk = SimulateGK(100, [3, 1, 2, 0.5])


# In[8]:

## TAR curve for g and k distribution
epsilon_seq = np.linspace(start=0, stop=5000, num = 50)
n = 200
k = 20
counter = 0
accepted_ratio = []
output_list = []
#Run abc with these epsilon and get the ratio of accepted samples

for eps in epsilon_seq:
    print(float(counter/len(epsilon_seq))*100,"%")
    counter += 1
    aux = 0
    for rep in range(k):
        ABC_run = ABC(GKPriorSampler, GKLiklihoodSimulator, GKSummary, eps, data_gk , n)
        output_dict = {'eps': eps, 'accept ratio':sum(ABC_run.accept)/float(n)}
        output_list.append(output_dict)
        aux=aux+sum(ABC_run.accept)/n
    accepted_ratio.append(aux/k)
    
print("DONE :D")
    
pickle.dump(output_list, open( "data/acceptance_rate_gk.p", "wb" ) )


# In[9]:

print(accepted_ratio)
plt.plot(epsilon_seq, accepted_ratio, '-')
plt.xlabel(r'$\epsilon$', fontsize=18)
plt.ylabel('Acceptance rate', fontsize=18)
plt.savefig('plots/TAR_curve_gk_distribution.pdf')
plt.show()


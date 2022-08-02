# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:07:09 2022

@author: 山抹微云
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from BenchmarkFunc.cec2017.functions import *
from BenchmarkFunc.unimodal.functions import *
from Application.nonlinear.problems import *
import time
    



from Algorithms.ESOA import ESOA
from Algorithms.GWO  import GWO
from Algorithms.MVO  import MVO
from Algorithms.SSA  import SSA
from Algorithms.PSO  import PSO
from Algorithms.HHO  import HHO
from Algorithms.GA   import GA
from Algorithms.SA   import SACauchy
from Algorithms.DE   import DE


# In[] 
func     = F1
n_dim = 30
lb   = np.ones(n_dim)*-100
ub   = np.ones(n_dim)*100

size_pop = 50
max_iter = 500

# In[]

print("###############################################")
print('             Optimization Process              ')
print("###############################################")

s = time.time()
esoa = ESOA(func, n_dim, size_pop, max_iter, lb, ub)
esoa.run()
x_hist_esoa, y_hist_esoa = esoa.x_hist_best, esoa.y_hist
e = time.time()
print("###############################################")
print('                                               ')
print('             ESOA cost time:{}                 '.format(e-s))
print('             Best Value: {}                 '.format(esoa.y_global_best))
print('                                               ')
print("###############################################")

s = time.time()
x_hist_gwo, y_hist_gwo = GWO(func, n_dim, size_pop, max_iter, lb, ub)
e = time.time()
print("###############################################")
print('                                               ')
print('             GWO  cost time:{}                 '.format(e-s))
print('             Best Value: {}                 '.format(y_hist_gwo[-1]))
print('                                               ')
print("###############################################")

s = time.time()
x_hist_mvo, y_hist_mvo = MVO(func, n_dim, size_pop, max_iter, lb, ub)
e = time.time()
print("###############################################")
print('                                               ')
print('             MVO  cost time:{}                 '.format(e-s))
print('             Best Value: {}                 '.format(y_hist_mvo[-1]))
print('                                               ')
print("###############################################")

s = time.time()
x_hist_ssa, y_hist_ssa = SSA(func, n_dim, size_pop, max_iter, lb, ub)
e = time.time()
print("###############################################")
print('                                               ')
print('             SSA  cost time:{}                 '.format(e-s))
print('             Best Value: {}                 '.format(y_hist_ssa[-1]))
print('                                               ')
print("###############################################")

s = time.time()
x_hist_pso, y_hist_pso = PSO(func, n_dim, size_pop, max_iter, lb, ub)
e = time.time()
print("###############################################")
print('                                               ')
print('             PSO  cost time:{}                 '.format(e-s))
print('             Best Value: {}                 '.format(y_hist_pso[-1]))
print('                                               ')
print("###############################################")

s = time.time()
x_hist_hho, y_hist_hho = HHO(func, n_dim, size_pop, max_iter, lb, ub)
e = time.time()
print("###############################################")
print('                                               ')
print('             HHO  cost time:{}                 '.format(e-s))
print('             Best Value: {}                 '.format(y_hist_hho[-1]))
print('                                               ')
print("###############################################")

s = time.time()
x_hist_ga, y_hist_ga = GA(func, n_dim, size_pop, max_iter, lb, ub)
e = time.time()
print("###############################################")
print('                                               ')
print('             GA   cost time:{}                 '.format(e-s))
print('             Best Value: {}                 '.format(y_hist_ga[-1]))
print('                                               ')
print("###############################################")

s = time.time()
x_hist_de, y_hist_de = DE(func, n_dim, size_pop, max_iter, lb, ub)
e = time.time()
print("###############################################")
print('                                               ')
print('             DE   cost time:{}                 '.format(e-s))
print('             Best Value: {}                 '.format(y_hist_de[-1]))
print('                                               ')
print("###############################################")

s = time.time()
sa = SACauchy(func, n_dim, max_iter, lb, ub)
sa.run()
x_hist_sa, y_hist_sa = sa.best_x_history, sa.best_y_history
e = time.time()
print("###############################################")
print('                                               ')
print('             SA  cost time:{}                 '.format(e-s))
print('             Best Value: {}                 '.format(y_hist_sa[-1]))
print('                                               ')
print("###############################################")





fig = plt.figure(figsize=(12,9))

sb.set_theme(style="ticks")
ax = fig.gca()

ax.plot(np.arange(500), y_hist_esoa,  color='#F75000', alpha=0.8, label='ESOA', linestyle='--')
ax.plot(np.arange(500), y_hist_gwo ,  color='#8600FF', alpha=0.8, label='GWO' , linestyle='--')
ax.plot(np.arange(500), y_hist_hho ,  color='#2894FF', alpha=0.8, label='HHO' , linestyle='--')
ax.plot(np.arange(500), y_hist_pso ,  color='#00CACA', alpha=0.8, label='PSO' , linestyle='--')
ax.plot(np.arange(500), y_hist_ga  ,  color='#8080C0', alpha=0.8, label='GA'  , linestyle='--')
ax.plot(np.arange(500), y_hist_mvo ,  color='#EE82EE', alpha=0.8, label='MVO'  , linestyle='--')
ax.plot(np.arange(500), y_hist_ssa ,  color='#D2A2CC', alpha=0.8, label='SSA' , linestyle='--')
#ax.plot(np.arange(500), y_hist_sa  ,  color='#D2A2CC', alpha=0.8, label='SA'  , linestyle='--')
ax.legend(loc="upper right", fontsize=24)
ax.set_xlabel('Iteration', fontsize=26)
ax.set_ylabel('Fitness', fontsize=26)

"""
    To Set The Limitaion Of Y-Axis
"""

#ax.set_ylim([-10, 1000])


plt.show()

# In[]






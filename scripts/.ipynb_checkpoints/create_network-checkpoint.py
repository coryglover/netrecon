### Script intended to refine reconstruction pipeline

import graph_tool.all as gt
from itertools import combinations
import numpy as np
from scipy.stats import binom
import scipy
import networkx as nx
import netrd
import argparse
import os
import copy
from network_models import *

# Accept input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,default=None,help='Graph Model')
parser.add_argument('--N',default=1000,type=int,help='Number of nodes')
parser.add_argument('--avg_k',default=1,type=float,help='Average degree')
parser.add_argument('--alpha',default=None,type=float,help='Preferential attachment exponent')
parser.add_argument('--T',default=1,type=float,help='Temperature of PSO model')
parser.add_argument('--beta',default=1,type=float,help='Beta of PSO model')
parser.add_argument('--file',default=None,type=str,help='Directory to store data')
parser.add_argument('--iter_num',default=None,type=int,help='Iteration Number')

# Read in parameters
args = parser.parse_args()
model = args.model
N = args.N
avg_k = args.avg_k
alpha = args.alpha
T = args.T
beta = args.beta
file = args.file
iter_num = args.iter_num

# Create random network
if model == 'random':
    p = avg_k / N
    g = gt.random_graph(N, lambda: (1 if np.random.random() < p else 0), directed=False)
    try:
        os.mkdir(f'{file}/k_{int(avg_k)}')
    except:
        directory = True
    new_file = f'{file}/k_{int(avg_k)}/iter_{iter_num}'
    os.mkdir(new_file)
    
elif model == 'pa':
    m = int(avg_k // 2)
    g = nonlinear_pa(N,m,alpha)
    try:
        os.mkdir(f'{file}/alpha_{alpha}')
    except:
        directory = True
    new_file = f'{file}/alpha_{alpha}/iter_{iter_num}'
    os.mkdir(new_file)

elif model == 'pso':
    m = avg_k // 2
    g = pso(N, int(m), T, beta)
    try:
        os.mkdir(f'{file}/T_{T}_beta_{beta}')
    except:
        directory = True
    new_file = f'{file}/T_{T}_beta_{beta}/iter_{iter_num}'
    os.mkdir(new_file)

# Save network
g.save(f'{new_file}/g.gml')

# Save params
np.savetxt(f'{new_file}/params.txt',np.array([N,avg_k,m,alpha,T,beta]))

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
parser.add_argument('--alpha',default=None,type=str,help='Preferential attachment exponent')
parser.add_argument('--T',default=1,type=float,help='Temperature of PSO model')
parser.add_argument('--beta',default=1,type=float,help='Beta of PSO model')
parser.add_argument('--file',default=None,type=str,help='Directory to store data')

# Read in parameters
args = parser.parse_args()
model = args.model
N = args.N
avg_k = args.avg_k
m = args.m
alpha = args.alpha
T = args.T
beta = args.beta
file = args.file

# Create random network
if model == 'random':
    p = avg_k / N
    g = gt.random_graph(N, lambda: (1 if random() < p else 0), directed=False)

elif model == 'pa':
    m = avg_k // 2
    g = gt.price_network(N, m, exp = alpha, directed=False)

elif model == 'pso':
    m = avg_k // 2
    g = pso(N, m, T, beta)

# Save network
g.save(f'{file}/g.gml')

# Save params
np.savetxt(f'{file}/params.txt',np.array([N,avg_k,m,alpha,T,beta]))

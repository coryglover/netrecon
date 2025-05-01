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

def star_network(n,num_nodes=50):
    """
    Creates a star network with graph-tool.
    
    Parameters:
        n (int): Number of nodes (including the central node).
    
    Returns:
        g (Graph): The generated star network.
    """
    g = gt.Graph(directed=False)

    # Add vertices
    vertices = [g.add_vertex() for _ in range(num_nodes)]

    # Central node (first node) connects to all others
    center = vertices[0]
    for i in range(1, n):
        g.add_edge(center, vertices[i])

    return g

def simulate(g,M=1000,dynamics='ising',beta=1.0,gamma=1.0):
    """
    Simulate Ising dynamics on network
    """
    # Run dynamics on network
    E = g.num_edges()
    N = g.num_vertices()
    w = g.new_ep("double", vals=N/(2*E))  # True edge weights, with very small noise

    if dynamics=='ising':
        istate = gt.IsingGlauberState(g, w=w,h = 0.0, beta = beta)
    else:
        istate = gt.SISState(g,beta=beta,gamma=gamma)
    X = []
    for m in range(M):
        istate.iterate_sync() # perform a single update of the dynamics
        X.append(istate.get_state().a.copy())
    X = np.array(X).T

    return X

def reconstruct(X, max_iters = 1000,dynamics = 'ising',beta=1.0):
    """
    Reconstruct network using dynamics X.
    """
    if dynamics == 'ising':
        state = gt.IsingGlauberBlockState(s=X,g=None,beta=beta)
    else:
        state = gt.EpidemicsBlockState(s=X,g=None)
    
    # initialize niter as one order of magnitude less than the number of nodes
    niter = len(X) / 10
    
    u_cur = None
    w_cur = None
    t_cur = None
    
    delta, _ = state.mcmc_sweep(niter=niter)
    entropy = state.entropy()
    for i in range(max_iters):
        delta, _ = state.mcmc_sweep(niter=1)
        if state.entropy() < entropy:
            entropy = state.entropy()
        else:
            continue
        
    u_cur = copy.deepcopy(state.get_graph())
    w_cur = copy.deepcopy(state.get_x())
    t_cur = copy.deepcopy(state.get_theta())

    return u_cur, w_cur, t_cur, entropy

parser = argparse.ArgumentParser()
parser.add_argument('--N',type=int,default=2,help='Minimum Number of Nodes')
# parser.add_argument('--max_N',type=int,default=100,help='Maximum Number of Nodes')
parser.add_argument('--min_steps',type=int,default=1,help='Minimum Number of Steps')
parser.add_argument('--max_steps',type=int,default=3,help='Maximum Number of Steps')
parser.add_argument('--iterations',type=int,default=100,help='Number of iterations')
parser.add_argument('--intervals',type=int,default=7,help='Number of intervals of Steps')
parser.add_argument('--dynamics',type=str,default='ising',help='Type of dynamics')
parser.add_argument('--beta',type=float,default=1.0,help='Beta')
parser.add_argument('--gamma',type=float,default=.2,help='Gamma')
parser.add_argument('--file',default=None,type=str,help='Directory to store data')

# Read in parameters
args = parser.parse_args()
N = args.N
min_steps = args.min_steps
max_steps = args.max_steps
intervals = args.intervals
dynamics = args.dynamics
iterations = args.iterations
beta = args.beta
gamma = args.gamma
file = args.file

try:
    os.mkdir(file)
except:
    exists = True

# Initialize array to save data
edge_array = np.zeros((N-1,intervals,iterations))
time_steps = np.logspace(min_steps,max_steps,intervals,dtype=int)
# correct_edges = np.zeros((intervals,iteration))
incorrect_edges = np.zeros((intervals,iteration))

# Create network
g = star_network(N)

for _ in range(iterations):

    # Simulate data
    X = simulate(g, int(10**max_steps), dynamics, beta, gamma)

    # Save simulation data
    np.savetxt(f'{file}/X_{_}.txt',X)

    # Reconstruct network with varying amounts of data
    for j, time in enumerate(time_steps):
        recon_g, w_r, t_r, entropy = reconstruct(X[:,:time],dynamics=dynamics,beta=beta)
        
        # Check that edges exist
        for i in range(1,N):
            if recon_g.edge(0,i) is not None:
                edge_array[i-1,j,_] = 1
        incorrect_edges[j,_] = recon_g.number_of_edges() - edge_array[:,j,_].sum()

# Save edge array
np.save(f'{file}/link_data.npy',edge_array)
np.save(f'{file}/incorrect_edges.npy',incorrect_edges)

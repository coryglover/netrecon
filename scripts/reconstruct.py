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

### Helper functions

def simulate(g,M=1000,dynamics='ising',beta=1.0,mu=1.0):
    """
    Simulate Ising dynamics on network
    """
    # Run dynamics on network
    E = g.num_edges()
    N = g.num_vertices()
    w = g.new_ep("double", vals=N/(2*E))  # True edge weights, with very small noise

    if dynamics=='ising':
        istate = gt.IsingGlauberState(g, w=w,h = 0.0, beta = beta)

    X = []
    for m in range(M):
        istate.iterate_sync() # perform a single update of the dynamics
        X.append(istate.get_state().a.copy())
    X = np.array(X).T

    return X

def reconstruct(X, max_iters = 1000,dynamics = 'ising',beta=1.0,mu=1.0):
    """
    Reconstruct network using dynamics X.
    """
    if dynamics == 'ising':
        state = gt.IsingGlauberBlockState(s=X,g=None,beta=beta)
    
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

def analyze_network(g):
    """
    Function to perform analysis of reconstruction for standard network statistics.
    
    Parameters:
        g (gt.Graph)
    
    Returns:
        avg_k (float) - average degree
        second_moment (float) - second moment of degree distribution
        c (float) - average clustering coefficient
        assortativity (float) - assortativity coefficient
        assort_var (float) - assortativity variance
    """
    # Get degree distribution
    deg = g.get_total_degrees(g.get_vertices())
    avg_k = np.mean(deg)
    second_moment = np.mean([k**2 for k in deg])
    kmax = np.max(deg)
    # Calculate average local clustering coefficient
    A = np.array(gt.adjacency(g).todense())
    c = nx.average_clustering(nx.from_numpy_array(A))
    
    # Calculate assortativity coefficient
    assortativity, assort_var = gt.assortativity(g,'total')
    
    
    # try:
    #     gamma_kern, gamma_hill = find_gamma(nx.from_numpy_array(A),path)
    # except:
    # gamma_kern = np.nan
    # gamma_hill = np.nan
    
    return avg_k, second_moment,kmax,  c, assortativity, assort_var

def calc_distances(g1,g2):
    """
    Calculate Jaccard, Hamming and Frobenius distance between two networks
    
    Parameters:
        g1 (nx.Graph)
        g2 (nx.Graph)
        
    Returns:
        jaccard (float)
        hamming (float)
        frobenius (float)
    """
    distances = {
    'Jaccard':                 netrd.distance.JaccardDistance(),
    'Hamming':                 netrd.distance.Hamming(),
    # 'HammingIpsenMikhailov':   netrd.distance.HammingIpsenMikhailov(),
    'Frobenius':               netrd.distance.Frobenius()}
    # 'PolynomialDissimilarity': netrd.distance.PolynomialDissimilarity(),
    # 'DegreeDivergence':        netrd.distance.DegreeDivergence(),
    # 'PortraitDivergence':      netrd.distance.PortraitDivergence(),
    # 'QuantumJSD':              netrd.distance.QuantumJSD(),
    # 'CommunicabilityJSD':      netrd.distance.CommunicabilityJSD(),
    # 'GraphDiffusion':          netrd.distance.GraphDiffusion(),
    # # 'ResistancePerturbation':  netrd.distance.ResistancePerturbation(),
    # 'NetLSD':                  netrd.distance.NetLSD(),
    # 'IpsenMikhailov':          netrd.distance.IpsenMikhailov(),
    # # 'NonBacktrackingSpectral': netrd.distance.NonBacktrackingSpectral(),
    # # 'DistributionalNBD':       netrd.distance.DistributionalNBD(),
    # # 'DMeasure':       netrd.distance.DMeasure(),
    # 'DeltaCon':                netrd.distance.DeltaCon(),
    # 'NetSimile':               netrd.distance.NetSimile()}
    dist = []
    for d_lab, d_i in distances.items():
        dist.append(d_i.dist(g1,g2))
    return dist[0], dist[1], dist[2]

# Accept input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dynamics',type=str,default=None,help='dynamics')
parser.add_argument('--beta',default=1.0,type=float,help='Beta parameter for Ising and SIS')
parser.add_argument('--mu',default=np.nan,type=float,help='Mu parameter for SIS')
parser.add_argument('--steps',default=1000,type=int,help='Number of steps to take')
parser.add_argument('--file',default=None,type=str,help='Directory storing data')

# Read in parameters
args = parser.parse_args()
dynamics = args.dynamics
beta = args.beta
mu = args.mu
steps = args.steps
file = args.file

# Get original graph from file
g = gt.load_graph(f'{file}/g.gml')

# Simulate dynamics
X = simulate(g,M=steps,dynamics=dynamics,beta=beta,mu=mu)

# Save dynamics
np.savetxt(f'{file}/{dynamics}_X_{steps}.txt')

# Try to reconstruct network
recon_g = reconstruct(X,dynamics=dynamics,beta=beta,mu=mu)

# Save network reconstruction
recon_g.save(f'{file}/recon_g_{steps}.gml')

# Analyze networks
g_stats = analyze_network(g)
recon_g_stats = analyze_network(recon_g)

# Get network distance
distances = calc_distances(g,recon_g)

# Save stats
np.savetxt(f'{file}/g_stats.txt',g_stats)
np.savetxt(f'{file}/recon_g_{steps}_stats.txt',recon_g_stats)
np.savetxt(f'{file}/distances_{steps}.txt',distances)
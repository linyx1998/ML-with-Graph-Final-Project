import numpy as np
import scipy as sp
import scipy.sparse, scipy.io, scipy.optimize
from scipy.special import expit
from scipy.sparse.linalg import eigsh
import argparse
import os
import random

import matplotlib.pyplot as plt
import matplotlib as mpl
import time

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', # Color blind color cycle
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=CB_color_cycle) 

clip_01 = lambda M : np.clip(M, a_min=0, a_max=1)

from DataPreprocess import *
# adj = read_cag_fb_wiki(ca_grqc_path="./data/CA-GrQc.txt")
# adj = read_cora(content_path="./data/cora/cora.content", cites_path="./data/cora/cora.cites")
adj = read_pubmed(folder_path="./data/Pubmed/data/")

def randomize_adjacency_matrix(adjacency_matrix, num_swaps):
    """Edge swap randomization preserving degree distribution"""
    if hasattr(adjacency_matrix, 'toarray'):
        matrix = adjacency_matrix.toarray()
    else:
        matrix = np.array(adjacency_matrix, copy=True)
    
    edges = np.argwhere(np.triu(matrix, k=1) == 1)
    num_edges = len(edges)
    
    for _ in range(num_swaps):
        idx1, idx2 = random.sample(range(num_edges), 2)
        
        u, v = edges[idx1]
        x, y = edges[idx2]
        
        if len({u, v, x, y}) < 4:
            continue
            
        if random.random() > 0.5:
            x, y = y, x
            
        a, b = sorted((u, x))
        c, d = sorted((v, y))
        
        if matrix[a, b] == 0 and matrix[c, d] == 0:
            matrix[u, v] = matrix[v, u] = 0
            matrix[x, y] = matrix[y, x] = 0
            
            matrix[a, b] = matrix[b, a] = 1
            matrix[c, d] = matrix[d, c] = 1
            
            edges[idx1] = [a, b]
            edges[idx2] = [c, d]
            
    return matrix

def create_er_graph(adjacency_matrix):
    """Create Erdős-Rényi random graph with same number of nodes and edges"""
    if hasattr(adjacency_matrix, 'toarray'):
        matrix = adjacency_matrix.toarray()
    else:
        matrix = np.array(adjacency_matrix)

    num_nodes = matrix.shape[0]
    num_edges = np.count_nonzero(np.triu(matrix, k=1))
    
    er_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    generated_edges = set()
    
    while len(generated_edges) < num_edges:
        needed = num_edges - len(generated_edges)
        batch_size = int(needed * 2) + 100
        
        sources = np.random.randint(0, num_nodes, batch_size)
        targets = np.random.randint(0, num_nodes, batch_size)
        
        mask = sources < targets
        sources = sources[mask]
        targets = targets[mask]
        
        for u, v in zip(sources, targets):
            if len(generated_edges) >= num_edges:
                break
            
            if (u, v) not in generated_edges:
                generated_edges.add((u, v))
                er_matrix[u, v] = 1
                er_matrix[v, u] = 1
                
    return er_matrix

def lpca_loss(factors, adj_s, rank): # adj_s = shifted adj with -1's and +1's
    n_row, n_col = adj_s.shape
    U = factors[:n_row*rank].reshape(n_row, rank)
    V = factors[n_row*rank:].reshape(rank, n_col)
    logits = U @ V
    prob_wrong = expit(-logits * adj_s)
    loss = (np.logaddexp(0,-logits*adj_s)).sum()# / n_element    
    U_grad = -(prob_wrong * adj_s) @ V.T# / n_element
    V_grad = -U.T @ (prob_wrong * adj_s)# / n_element
    print(loss)
    return loss, np.concatenate((U_grad.flatten(), V_grad.flatten()))

save_interval = 100 # the number of iterations after which to save the embeddings

def callback_recm(x_i): # prints the loss and periodically saves the factors
    global iter_num
    iter_num += 1
    if iter_num % save_interval == 0:
        global factors, n_row, n_col, rank, adj
        factors = x_i
        U = factors[:n_row*rank].reshape(n_row, rank)
        V = factors[n_row*rank:].reshape(rank, n_col)
        frob_error_norm = np.linalg.norm(clip_01(U@V) - adj) / sp.sparse.linalg.norm(adj)
        print(iter_num, "Frob_error_norm: ", frob_error_norm)
        
# generate a rank k TSVD factorization of a small sparse matrix adj
def factor_TSVD(adj, k):
    w, v = np.linalg.eigh(np.array(adj.todense()))
    order = np.argsort(np.abs(w))[::-1]
    w = w[order[:k]]
    v = v[:,order[:k]]
    U_tsvd, V_tsvd = v * np.sqrt(np.abs(w))[None,:], (np.sign(w)*np.sqrt(np.abs(w)))[:,None] * v.T
    return U_tsvd, V_tsvd

# compute number of triangles in the subgraph induced by the first i nodes
def get_num_tri(i):
    global adj_recon_sort
    sub_adj = adj_recon_sort[:i+1,:i+1]
    num_tri = (1. * sub_adj @ sub_adj @ sub_adj).diagonal().sum() / 6
    return num_tri

def lpca_fit(adj, rank_value, maxiter=2000):
    global iter_num, factors, n_row, n_col, rank, frob_error_norm
    rank = rank_value
    n_row, n_col = adj.shape
    factors = -1+2*np.random.random(size=(np.sum(adj.shape)*rank)) # initalize uniformly on [-1,+1]
    iter_num = 0
    U = factors[:n_row*rank].reshape(n_row, rank)
    V = factors[n_row*rank:].reshape(rank, n_col)
    
    res = scipy.optimize.minimize(lpca_loss, x0=factors, 
                              args=(-1 + 2*np.array(adj.todense()), rank), jac=True, method='L-BFGS-B', 
                              callback=callback_recm, 
                              options={'maxiter':maxiter}
                             )
    factors = res.x
    U = res.x[:n_row*rank].reshape(n_row, rank)
    V = res.x[n_row*rank:].reshape(rank, n_col)
    frob_error_norm = np.linalg.norm(1.*(U @ V > 0) - adj) / sp.sparse.linalg.norm(adj)
    print("LPCA Frob norm error: ", frob_error_norm)
    return U, V

def tsvd_fit(adj, rank):
    U_tsvd, V_tsvd = factor_TSVD(adj, rank)
    frob_error_norm = np.linalg.norm(clip_01(U_tsvd@V_tsvd) - adj) / sp.sparse.linalg.norm(adj)
    print("TSVD Frob norm error: ", frob_error_norm)
    return U_tsvd, V_tsvd

def save_matrices(U, V, output_dir, prefix="lpca"):
    """Save U and V matrices to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, f"U_{prefix}.npy"), U)
    np.save(os.path.join(output_dir, f"V_{prefix}.npy"), V)
    
    print(f"Matrices saved to {output_dir}/:")
    print(f"  U_{prefix}.npy (shape: {U.shape})")
    print(f"  V_{prefix}.npy (shape: {V.shape})")

def compute_network_stats(adj):
    """Compute basic network statistics"""
    if hasattr(adj, 'toarray'):
        adj_array = adj.toarray()
    else:
        adj_array = np.array(adj)
    
    n = adj_array.shape[0]
    m = np.sum(adj_array) / 2
    mean_degree = 2 * m / n
    
    print(f"\nNetwork Statistics:")
    print(f"  Nodes: {n}")
    print(f"  Edges: {int(m)}")
    print(f"  Mean degree: {mean_degree:.4f}")
    
    return n, m, mean_degree


def main():
    global adj  # Make adj global for callback function
    
    parser = argparse.ArgumentParser(description='LPCA/TSVD Matrix Factorization')
    
    # Algorithm selection
    parser.add_argument('--method', type=str, default='lpca', choices=['lpca', 'tsvd'])
    
    # Graph modification
    parser.add_argument('--graph_type', type=str, default='original',
                        choices=['original', 'edge_swap', 'erdos_renyi'])
    parser.add_argument('--swap_multiplier', type=int, default=50)
    
    # Algorithm parameters
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--maxiter', type=int, default=2000)
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./matrices/')
    parser.add_argument('--prefix', type=str, default=None)
    
    # Other options
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--stats', action='store_true')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    if args.stats:
        n, m, mean_degree = compute_network_stats(adj)
    
    if args.graph_type == 'edge_swap':
        if hasattr(adj, 'toarray'):
            m = np.sum(adj.toarray()) / 2
        else:
            m = np.sum(adj) / 2
        num_swaps = int(args.swap_multiplier * m)
        adj = sp.sparse.csr_matrix(randomize_adjacency_matrix(adj, num_swaps))
    elif args.graph_type == 'erdos_renyi':
        adj = sp.sparse.csr_matrix(create_er_graph(adj))
    
    # Run factorization
    if args.method == 'lpca':
        U, V = lpca_fit(adj, rank_value=args.rank, maxiter=args.maxiter)
    else:
        U, V = tsvd_fit(adj, rank=args.rank)
    
    # Generate prefix for output files
    if args.prefix is None:
        if args.graph_type == 'original':
            prefix = f"{args.method}_rank{args.rank}"
        else:
            prefix = f"{args.method}_{args.graph_type}_rank{args.rank}"
    else:
        prefix = args.prefix
    
    # Save matrices
    save_matrices(U, V, args.output_dir, prefix=prefix)

if __name__ == "__main__":
    main()
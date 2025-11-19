import numpy as np
import scipy as sp
from scipy.sparse.csgraph import connected_components


def read_cora(content_path="./cora/cora.content", cites_path="./cora/cora.cites"):
    content = np.genfromtxt(content_path, dtype=str)
    node_ids = content[:, 0]
    id_map = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    edges = np.genfromtxt(cites_path, dtype=str)
    rows = []
    cols = []

    for src, dst in edges:
        if src in id_map and dst in id_map:
            rows.append(id_map[src])
            cols.append(id_map[dst])

    adj = sp.sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n, n),
        dtype=np.float32
    )

    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj = adj.tocsr()

    print("Before GC:", adj.shape, "nnz =", adj.nnz)
    n_comp, labels = connected_components(adj)
    giant = np.argmax(np.bincount(labels))
    keep = np.where(labels == giant)[0]

    adj = adj[keep][:, keep].tocsr()
    n = adj.shape[0]
    print("After GC:", adj.shape, "nnz =", adj.nnz)
    
    return adj


def read_CAHepPh(ca_hepph_path="./CA-HepPh.txt"):
    edges = []
    with open(ca_hepph_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = map(int, line.split())
            edges.append((u, v))

    edges = [(u, v) for (u, v) in edges if u != v]

    nodes = sorted({u for e in edges for u in e})
    id2idx = {nid: i for i, nid in enumerate(nodes)}
    rows = [id2idx[u] for (u, v) in edges]
    cols = [id2idx[v] for (u, v) in edges]

    n_all = len(nodes)
    adj_full = sp.sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_all, n_all))

    adj_full = adj_full + adj_full.T
    adj_full.data[:] = 1.0
    adj_full = adj_full.tocsr()

    print("Before GC:", adj_full.shape, "nnz =", adj_full.nnz)
    n_comp, labels = connected_components(adj_full)
    giant = np.argmax(np.bincount(labels))
    keep = np.where(labels == giant)[0]

    adj = adj_full[keep][:, keep].tocsr()
    n = adj.shape[0]
    print("After GC:", adj.shape, "nnz =", adj.nnz)
    
    return adj


def read_CAGrQc(ca_grqc_path="./CA-GrQc.txt"):
    edges = []
    with open(ca_grqc_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = map(int, line.split())
            edges.append((u, v))

    edges = [(u, v) for (u, v) in edges if u != v]

    nodes = sorted({u for e in edges for u in e})
    id2idx = {nid: i for i, nid in enumerate(nodes)}
    rows = [id2idx[u] for (u, v) in edges]
    cols = [id2idx[v] for (u, v) in edges]

    n_all = len(nodes)
    adj_full = sp.sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_all, n_all))

    adj_full = adj_full + adj_full.T
    adj_full.data[:] = 1.0
    adj_full = adj_full.tocsr()
    print("Shape:", adj_full.shape, "nnz =", adj_full.nnz)

    return adj_full
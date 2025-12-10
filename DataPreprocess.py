import numpy as np
import scipy as sp
from scipy.sparse.csgraph import connected_components
import os


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
    
    mean_deg = adj.sum(axis=1).A1.mean()
    print("Mean degree =", mean_deg)
    
    return adj


def read_cahepph(ca_hepph_path="./CA-HepPh.txt"):
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
    
    mean_deg = adj.sum(axis=1).A1.mean()
    print("Mean degree =", mean_deg)
    
    return adj


def read_cag_fb_wiki(ca_grqc_path="./CA-GrQc.txt"):
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
    adj = sp.sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_all, n_all))

    adj = adj + adj.T
    adj.data[:] = 1.0
    adj = adj.tocsr()
    print("Shape:", adj.shape, "nnz =", adj.nnz)
    
    mean_deg = adj.sum(axis=1).A1.mean()
    print("Mean degree =", mean_deg)

    return adj


def read_pubmed(folder_path):
    node_path = os.path.join(folder_path, "Pubmed-Diabetes.NODE.paper.tab")
    cite_path = os.path.join(folder_path, "Pubmed-Diabetes.DIRECTED.cites.tab")

    with open(node_path, "r") as f:
        lines = f.readlines()

    data_lines = lines[2:]
    node_ids = []

    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if not parts:
            continue
        node_ids.append(parts[0])

    node_ids = np.array(node_ids, dtype=str)
    N = len(node_ids)
    id_map = {pid: i for i, pid in enumerate(node_ids)}

    rows, cols = [], []

    with open(cite_path, "r") as f:
        cite_lines = f.readlines()[2:]

    for line in cite_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        if ":" not in parts[1] or ":" not in parts[3]:
            continue
        src = parts[1].split(":", 1)[1]
        dst = parts[3].split(":", 1)[1]
        if src in id_map and dst in id_map:
            rows.append(id_map[src])
            cols.append(id_map[dst])

    adj = sp.sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj = adj.tocsr()
    
    # n_comp, labels = connected_components(adj)
    # giant = np.argmax(np.bincount(labels))
    # keep = np.where(labels == giant)[0]

    # adj = adj[keep][:, keep].tocsr()

    mean_deg = adj.sum(axis=1).A1.mean()
    print("Nodes =", N)
    print("Mean degree =", mean_deg)

    return adj

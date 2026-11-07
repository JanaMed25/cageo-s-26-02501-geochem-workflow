"""Positive Pearson correlation network with Benjamini-Hochberg FDR control."""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr


def correlation_network(frame: pd.DataFrame, correlation_threshold=0.5, fdr_q=0.05):
    cols = list(frame.columns)
    records = []
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            pair = frame[[left, right]].dropna()
            if len(pair) < 4 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
                continue
            r, p = pearsonr(pair[left], pair[right])
            records.append({"source": left, "target": right, "r": float(r),
                            "p_value": float(p), "n": int(len(pair))})
    edges = pd.DataFrame(records, columns=["source", "target", "r", "p_value", "n"])
    if not edges.empty:
        order = np.argsort(edges["p_value"].to_numpy())
        p_sorted = edges["p_value"].to_numpy()[order]
        m = len(p_sorted)
        adjusted_sorted = np.minimum.accumulate((p_sorted * m / np.arange(1, m + 1))[::-1])[::-1]
        adjusted = np.empty(m)
        adjusted[order] = np.minimum(adjusted_sorted, 1.0)
        edges["q_value"] = adjusted
        edges = edges[(edges.r >= correlation_threshold) & (edges.q_value <= fdr_q)].copy()
    graph = nx.Graph()
    graph.add_nodes_from(cols)
    for row in edges.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=row.r, q_value=row.q_value, n=row.n)
    degree = dict(graph.degree())
    centrality = nx.degree_centrality(graph)
    communities = list(nx.community.greedy_modularity_communities(graph, weight="weight")) if graph.number_of_edges() else [{n} for n in graph]
    community_id = {node: idx + 1 for idx, group in enumerate(communities) for node in group}
    nodes = pd.DataFrame({
        "element": cols,
        "degree": [degree.get(c, 0) for c in cols],
        "degree_centrality": [centrality.get(c, 0.0) for c in cols],
        "community": [community_id.get(c, 0) for c in cols],
    })
    return nodes, edges.reset_index(drop=True), graph

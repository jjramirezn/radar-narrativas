"""Coordination: find groups of accounts that act together, and keep them apart from genuine crowds.

A burst of comments proves nothing (viral posts draw hundreds of genuine comments in
minutes). What gives an operation away is repetition: the same accounts, together, again
and again. So:
1. Events: each comment falls in a (post, time window) event, on two grids offset by half
   a window so bursts aren't cut in two, at two scales (15 and 60 minutes) because some
   operations fire within minutes and others spread over an hour.
2. Each account becomes a vector over events, weighted by rarity (IDF): sharing a quiet
   moment means much more than sharing a viral one, and crowded windows are ignored.
   Cosine similarity between accounts measures "acting together beyond chance".
3. Extra links: rare hashtags used by the same few accounts, and replying to each other
   repeatedly. (Shared phrases alone are useless: genuine users repeat each other a lot.)
4. Leiden finds groups. Members with fewer than two links inside their group are dropped,
   and only tight groups survive: random genuine links form loose chains, operations form
   dense webs.
"""
import re
from collections import defaultdict

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy import sparse

from . import text
from .context import Context

WINDOWS_MIN = (15, 60)
MAX_EVENT_ACCOUNTS = 60    # windows with more accounts than this are crowds, not signals
MIN_SHARED_EVENTS = 3      # a co-activity link needs this many shared events
MIN_COSINE = 0.25
RARE_HASHTAG_MAX_ACCOUNTS = 30
MIN_GROUP = 4
MIN_INTERNAL_LINKS = 2
MIN_DENSITY = 0.15
HASHTAG = re.compile(r"#\w+")


def event_matrix(df: pd.DataFrame, window_min: int) -> tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix]:
    """Accounts x events, IDF-weighted, rows L2-normalized; plus the plain presence matrix."""
    accounts, acc_idx = np.unique(df["cuenta_id"].to_numpy(), return_inverse=True)
    width = window_min * 60
    keys = []
    for offset in (0, width / 2):
        bucket = ((df["t"].to_numpy() + offset) // width).astype(np.int64)
        keys.append(df["publicacion_id"].astype(str).to_numpy() + f"|{offset}|" + bucket.astype(str))
    rows = np.concatenate([acc_idx, acc_idx])
    _, ev_idx = np.unique(np.concatenate(keys), return_inverse=True)
    m = sparse.csr_matrix((np.ones(len(rows)), (rows, ev_idx)), shape=(len(accounts), ev_idx.max() + 1))
    m.data[:] = 1.0  # presence, not count
    df_ev = np.asarray(m.sum(0)).ravel()
    idf = np.where(df_ev <= MAX_EVENT_ACCOUNTS, np.log(len(accounts) / np.maximum(df_ev, 1)), 0.0)
    m = m.multiply(idf).tocsr()
    m.eliminate_zeros()
    presence = (m > 0).astype(np.float32).tocsr()
    norms = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    m = (sparse.diags(1 / np.where(norms > 0, norms, 1)) @ m).tocsr()
    return m, accounts, presence


def coactivity_edges(df: pd.DataFrame, window_min: int) -> dict:
    m, accounts, presence = event_matrix(df, window_min)
    sim = sparse.triu(m @ m.T, k=1).tocsr()
    shared = sparse.triu(presence @ presence.T, k=1).tocsr()
    sim.sort_indices()
    shared.sort_indices()
    # Same sparsity pattern (same nonzero events), so the data arrays line up.
    keep = (sim.data >= MIN_COSINE) & (shared.data >= MIN_SHARED_EVENTS)
    coo = sim.tocoo()
    return {(accounts[i], accounts[j]): float(s) for i, j, s in zip(coo.row[keep], coo.col[keep], coo.data[keep])}


def hashtag_edges(df: pd.DataFrame) -> dict:
    """Accounts sharing hashtags that few accounts use, weighted by rarity."""
    users = defaultdict(set)
    for acc, raw in zip(df["cuenta_id"], df["texto"]):
        for tag in HASHTAG.findall(raw.lower()):
            users[text.strip_accents(tag).rstrip("_")].add(acc)
    edges = defaultdict(float)
    for accs in users.values():
        if 2 <= len(accs) <= RARE_HASHTAG_MAX_ACCOUNTS:
            accs = sorted(accs)
            w = 1.0 / np.log2(1 + len(accs))
            for a in range(len(accs)):
                for b in range(a + 1, len(accs)):
                    edges[(accs[a], accs[b])] += w
    return dict(edges)


def reply_edges(df: pd.DataFrame) -> dict:
    """Pairs of accounts that reply to each other at least twice."""
    owner = df.set_index("comentario_id")["cuenta_id"]
    r = df[df["respuesta_a"].notna()]
    pairs = pd.DataFrame({"a": r["cuenta_id"].to_numpy(), "b": r["respuesta_a"].map(owner).to_numpy()}).dropna()
    pairs = pairs[pairs["a"] != pairs["b"]]
    if pairs.empty:
        return {}
    lo, hi = np.minimum(pairs["a"], pairs["b"]), np.maximum(pairs["a"], pairs["b"])
    counts = pd.Series(1, index=pd.MultiIndex.from_arrays([lo, hi])).groupby(level=[0, 1]).sum()
    return {k: float(v) for k, v in counts.items() if v >= 2}


def build_graph(df: pd.DataFrame) -> tuple[ig.Graph, list, dict]:
    sources = [coactivity_edges(df, w) for w in WINDOWS_MIN]
    co = defaultdict(float)
    for edges in sources:
        for k, w in edges.items():
            co[k] = max(co[k], w)
    weights = defaultdict(float)
    for edges, scale in ((co, 1.0), (hashtag_edges(df), 0.5), (reply_edges(df), 0.1)):
        for k, w in edges.items():
            weights[k] += scale * w
    nodes = sorted({a for k in weights for a in k})
    pos = {a: i for i, a in enumerate(nodes)}
    graph = ig.Graph(n=len(nodes), edges=[(pos[a], pos[b]) for a, b in weights],
                     edge_attrs={"weight": list(weights.values())})
    counts = {"aristas_coactividad": len(co), "aristas_hashtag": len(hashtag_edges(df)),
              "aristas_respuestas": len(reply_edges(df))}
    return graph, nodes, counts


def run(ctx: Context) -> pd.DataFrame:
    df = ctx.comments
    graph, nodes, counts = build_graph(df)
    part = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition, weights="weight",
                                    seed=ctx.settings.seed)
    rows, assigned, webs = [], {}, {}
    for c, members in enumerate(part):
        members = list(members)
        # Peel off members with too few links inside the group (accounts merely touching it).
        while len(members) >= MIN_GROUP:
            sub = graph.induced_subgraph(members)
            weak = [members[i] for i, d in enumerate(sub.degree()) if d < MIN_INTERNAL_LINKS]
            if not weak:
                break
            members = [m for m in members if m not in set(weak)]
        if len(members) < MIN_GROUP:
            continue
        sub = graph.induced_subgraph(members)
        density = sub.density()
        rows.append({"comunidad": c, "cuentas": len(members), "densidad": round(density, 3),
                     "peso_medio": round(float(np.mean(sub.es["weight"])), 3), "coordinada": density >= MIN_DENSITY})
        if density >= MIN_DENSITY:
            for m in members:
                assigned[nodes[m]] = c
            webs[c] = (len(members), [e.tuple for e in sub.es], list(sub.es["weight"]))
    stats = pd.DataFrame(rows, columns=["comunidad", "cuentas", "densidad", "peso_medio", "coordinada"])
    order = stats.loc[stats["coordinada"].astype(bool)].sort_values("cuentas", ascending=False)["comunidad"].tolist()
    names = {c: f"op{i + 1:02d}" for i, c in enumerate(order)}
    groups = pd.DataFrame({"cuenta_id": list(assigned), "grupo": [names[c] for c in assigned.values()]},
                          dtype="string")
    if groups.empty:  # nothing coordinated found: still write a typed, empty table
        ctx.sql("CREATE OR REPLACE TABLE grupos (cuenta_id VARCHAR, grupo VARCHAR)")
    else:
        ctx.put_table("grupos", groups)
    ctx.report["coordinacion"] = {**counts, "comunidades_candidatas": len(stats),
                                  "operaciones": len(order), "cuentas_marcadas": len(groups)}
    ctx.coord_stats = stats
    # Network density per operation, shown in the dashboard as evidence for the flag.
    ctx.report["operaciones"] = {names[c]: {"densidad": float(d)}
                                 for c, d in zip(stats["comunidad"], stats["densidad"]) if c in names}
    # Each operation's internal links (member positions 0..n-1), drawn as a small web in the dashboard.
    ctx.coord_webs = {names[c]: web for c, web in webs.items() if c in names}
    return groups

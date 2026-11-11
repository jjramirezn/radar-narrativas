"""Narratives: find the claims that circulate and assign each comment to one, or to noise.

1. Embed unique normalized texts (cached).
2. Over-cluster them with k-means into many small, pure "micro-clusters".
3. Noise: short comments outside reply threads, and micro-clusters whose comments
   almost never take part in threads (off-topic chatter rarely gets answered).
4. Merge micro-clusters into narratives. Two micro-clusters belong together when their
   comments reply to each other (a reply stays within its parent's narrative, even when
   it argues back) or when they mean nearly the same thing. Leiden finds the groups.
5. Each reply thread votes on one narrative for all its comments.
6. A narrative's origin is the earliest burst of its most typical comments.
"""
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer

from . import embed, text
from .context import Context

SHORT_WORDS = 3            # unconnected comments this short are reactions or chatter
SHORT_REPLY_WORDS = 5      # replies this short just agree or disagree with their parent
UNITS_PER_MICRO_CLUSTER = 120.0  # unique texts per k-means micro-cluster, on average
NOISE_CONNECTED_RATIO = 0.3  # a micro-cluster is noise below this share of the typical connected rate
MIN_REPLY_RATE = 0.02      # below this reply rate, reply-based signals are skipped
ORIGIN_BURST = 3           # origin = first time this many typical comments fall within ORIGIN_WINDOW_H
ORIGIN_WINDOW_H = 6.0
STOPWORDS = set("""a al algo ante con como contra cual de del desde donde el ella ellos en entre era es esa ese
eso esta este esto estos fue ha hay la las le les lo los mas me mi muy nada ni no nos o para pero por que
se si sin sobre son su sus te ti tiene todo todos tu un una uno unos y ya yo aca ahi asi che posta bueno
mira ver digo decirlo hay que nadie dice lo pasa verdad siempre igual tal cual""".split())


def load_comments(ctx: Context) -> pd.DataFrame:
    df = ctx.con.execute("""
        SELECT comentario_id, publicacion_id, cuenta_id, epoch(ts) AS t, texto, respuesta_a
        FROM comentarios ORDER BY orden
    """).df()
    df["emb_text"] = df["texto"].map(text.for_embedding)
    df["nw"] = df["emb_text"].str.split().str.len().fillna(0).astype(int)
    parents = set(df["respuesta_a"].dropna())
    df["conectado"] = df["respuesta_a"].notna() | df["comentario_id"].isin(parents)
    return df


def thread_roots(df: pd.DataFrame) -> pd.Series:
    """Index of each comment's thread root (cycles and missing parents stop the walk)."""
    pos = pd.Series(np.arange(len(df)), index=df["comentario_id"])
    parent = df["respuesta_a"].map(pos).fillna(-1).astype(int).to_numpy()
    root = np.arange(len(df))
    for _ in range(50):
        p = parent[root]
        if not (p >= 0).any():
            break
        root = np.where(p >= 0, p, root)
    return pd.Series(root, index=df.index)


def micro_clusters(X: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    k = int(np.clip(len(X) / UNITS_PER_MICRO_CLUSTER, 40, 1000))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=8192, n_init=3, max_iter=200)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    return labels, centroids


def reply_lift(ids: np.ndarray, reply_pairs: np.ndarray) -> dict:
    """For each pair of micro-clusters: (reply links between them, links / links expected by chance).

    Expected = e_a * e_b / E, with e the reply endpoints in each micro-cluster and E all endpoints.
    Inside one narrative the lift is about the number of narratives; across narratives it's near 0.
    That ratio doesn't depend on dataset size, unlike raw counts.
    """
    pos = {m: i for i, m in enumerate(ids)}
    a = np.array([pos[x] for x in reply_pairs[:, 0]])
    b = np.array([pos[x] for x in reply_pairs[:, 1]])
    endpoints = np.bincount(np.concatenate([a, b]), minlength=len(ids)).astype(float)
    total = endpoints.sum()
    counts = pd.Series(1, index=pd.MultiIndex.from_arrays([np.minimum(a, b), np.maximum(a, b)])).groupby(level=[0, 1]).sum()
    return {(i, j): (c, c / (endpoints[i] * endpoints[j] / total)) for (i, j), c in counts.items() if i != j}


def merge(mc_stats: pd.DataFrame, reply_pairs: np.ndarray, centroids: np.ndarray, seed: int,
          sim_floor: float = 0.95, resolution: float = 1.0, method: str = "lift",
          min_links: int = 3, min_lift: float = 2.0) -> np.ndarray:
    """Group micro-clusters into narratives using reply links and near-identical meaning."""
    ids = mc_stats.index.to_numpy()
    n = len(ids)
    weights = {}
    if len(reply_pairs):
        if method == "lift":
            for key, (count, lift) in reply_lift(ids, reply_pairs).items():
                if count >= min_links and lift >= min_lift:
                    weights[key] = np.log(lift)
        else:
            pos = {m: i for i, m in enumerate(ids)}
            conn = mc_stats["conectados"].clip(lower=1).to_numpy()
            a = np.array([pos[x] for x in reply_pairs[:, 0]])
            b = np.array([pos[x] for x in reply_pairs[:, 1]])
            counts = pd.Series(1, index=pd.MultiIndex.from_arrays([np.minimum(a, b), np.maximum(a, b)])).groupby(level=[0, 1]).sum()
            for (i, j), c in counts.items():
                if i != j:
                    weights[(i, j)] = c / np.sqrt(conn[i] * conn[j])
    sims = centroids[ids] @ centroids[ids].T
    ii, jj = np.where(np.triu(sims, 1) > sim_floor)
    for i, j in zip(ii, jj):
        weights[(i, j)] = weights.get((i, j), 0.0) + (sims[i, j] - sim_floor) / (1 - sim_floor)
    graph = ig.Graph(n=n, edges=list(weights), edge_attrs={"weight": list(weights.values())})
    if method == "lift-components":
        return np.array(graph.connected_components().membership)
    part = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=resolution, seed=seed)
    return np.array(part.membership)


def key_phrases(docs: pd.Series, labels: pd.Series, top: int = 6) -> dict:
    """Class-based TF-IDF: words frequent in one narrative and rare in the others."""
    grouped = docs.groupby(labels).apply(lambda s: " ".join(s.sample(min(len(s), 3000), random_state=0)))
    vec = CountVectorizer(ngram_range=(1, 2), stop_words=list(STOPWORDS), token_pattern=r"(?u)\b[a-záéíóúüñ]{3,}\b")
    counts = vec.fit_transform(grouped.to_numpy()).toarray().astype(float)
    tf = counts / counts.sum(1, keepdims=True).clip(min=1)
    idf = np.log(1 + counts.sum() / counts.sum(0).clip(min=1))
    scores = tf * idf
    terms = np.array(vec.get_feature_names_out())
    return {label: list(terms[np.argsort(-row)[:top]]) for label, row in zip(grouped.index, scores)}


def origin(times: np.ndarray) -> float:
    times = np.sort(times)
    window = ORIGIN_WINDOW_H * 3600
    for i in range(len(times) - ORIGIN_BURST + 1):
        if times[i + ORIGIN_BURST - 1] - times[i] <= window:
            return float(times[i])
    return float(times[0])


def run(ctx: Context, sim_floor: float = 0.95, resolution: float = 1.0, method: str = "lift") -> pd.DataFrame:
    s = ctx.settings
    df = load_comments(ctx)
    codes, uniques = pd.factorize(df["emb_text"])
    df["u"] = codes
    X = embed.embed_texts(list(uniques), s.embedding_model, ctx.trabajo.parent / "_cache")

    reply_rate = df["respuesta_a"].notna().mean()
    use_replies = reply_rate >= MIN_REPLY_RATE
    df["ruido"] = (df["nw"] <= SHORT_WORDS) & (~df["conectado"] if use_replies else True)

    # Micro-clusters over the unique texts of candidate comments.
    # Short replies ("Tal cual", "Exacto") carry no topic of their own: they inherit their
    # parent's narrative at the end and stay out of clustering, where they'd link everything.
    df["respuesta_corta"] = df["respuesta_a"].notna() & (df["nw"] <= SHORT_REPLY_WORDS)
    clusterable = ~df["ruido"] & ~df["respuesta_corta"]
    cand_u = np.unique(df.loc[clusterable, "u"])
    mc_of_u = np.full(len(uniques), -1)
    labels, centroids = micro_clusters(X[cand_u], s.seed)
    mc_of_u[cand_u] = labels
    df["mc"] = np.where(clusterable, mc_of_u[df["u"]], -1)

    stats = df[df["mc"] >= 0].groupby("mc").agg(tamano=("mc", "size"), conectados=("conectado", "sum"),
                                                 palabras=("nw", "mean"))
    stats["tasa_conectados"] = stats["conectados"] / stats["tamano"]
    if use_replies:
        typical = np.average(stats["tasa_conectados"], weights=stats["tamano"])
        noise_mc = stats.index[stats["tasa_conectados"] < NOISE_CONNECTED_RATIO * typical]
        df.loc[df["mc"].isin(noise_mc), "ruido"] = True
        stats = stats.drop(noise_mc)
    df.loc[df["ruido"], "mc"] = -1

    # Reply links between surviving micro-clusters.
    pos = pd.Series(np.arange(len(df)), index=df["comentario_id"])
    child = df.index[df["respuesta_a"].notna()].to_numpy()
    parent = df.loc[child, "respuesta_a"].map(pos).to_numpy()
    mc = df["mc"].to_numpy()
    ok = (mc[child] >= 0) & (mc[parent] >= 0)
    pairs = np.stack([mc[child][ok], mc[parent][ok]], axis=1) if use_replies else np.empty((0, 2), int)

    membership = merge(stats, pairs, centroids, s.seed, sim_floor, resolution, method)
    narr_of_mc = pd.Series(membership, index=stats.index)
    df["grupo"] = df["mc"].map(narr_of_mc).fillna(-1).astype(int)

    # Each thread votes on one narrative; the comment that opened it counts double.
    # Short replies have no vote yet, then take their parent's narrative down the chain.
    if use_replies:
        df["raiz"] = thread_roots(df)
        in_thread = df["raiz"].map(df.groupby("raiz").size()) > 1
        voters = df.loc[in_thread & (df["grupo"] >= 0), ["raiz", "grupo"]].copy()
        voters["peso"] = np.where(voters.index == voters["raiz"], 2, 1)
        votes = (voters.groupby(["raiz", "grupo"])["peso"].sum().reset_index()
                 .sort_values(["raiz", "peso"], ascending=[True, False]))
        voted = votes.drop_duplicates("raiz").set_index("raiz")["grupo"]
        df.loc[in_thread, "grupo"] = df.loc[in_thread, "raiz"].map(voted).fillna(-1).astype(int)
        parent_pos = df["respuesta_a"].map(pos)
        short = df.index[df["respuesta_corta"]].to_numpy()
        for _ in range(20):
            inherited = df["grupo"].to_numpy()[parent_pos.iloc[short].astype(int).to_numpy()]
            changed = (df.loc[short, "grupo"].to_numpy() != inherited) & (inherited >= 0)
            if not changed.any():
                break
            df.loc[short[changed], "grupo"] = inherited[changed]

    # Name narratives by size: n01 is the largest.
    sizes = df.loc[df["grupo"] >= 0, "grupo"].value_counts()
    names = {g: f"n{i + 1:02d}" for i, g in enumerate(sizes.index)}
    df["narrativa"] = df["grupo"].map(names).fillna("")

    # Typical comments: closest to their micro-cluster's centroid, used for origins.
    sim = np.einsum("ij,ij->i", X[df["u"]], centroids[np.clip(df["mc"], 0, None)])
    df["tipicidad"] = np.where(df["mc"] >= 0, sim, -1.0)
    phrases = key_phrases(df.loc[df["narrativa"] != "", "emb_text"], df.loc[df["narrativa"] != "", "narrativa"])
    rows = []
    for name, g in df[df["narrativa"] != ""].groupby("narrativa"):
        core = g[g["tipicidad"] >= g["tipicidad"].median()]
        terms = phrases.get(name, [])
        rows.append({"narrativa": name, "titulo": " / ".join(terms[:3]), "frases_clave": "|".join(terms),
                     "origen_epoch": origin(core["t"].to_numpy() if len(core) >= ORIGIN_BURST else g["t"].to_numpy())})
    narr_df = pd.DataFrame(rows, columns=["narrativa", "titulo", "frases_clave", "origen_epoch"])
    ctx.put_table("narrativas", narr_df)
    ctx.report["narrativas"] = {
        "micro_clusters": int(len(centroids)), "micro_clusters_ruido": int(len(centroids) - len(stats)),
        "narrativas": len(narr_df), "comentarios_ruido": int((df["narrativa"] == "").sum()),
        "tasa_respuestas": round(float(reply_rate), 4),
    }
    ctx.comments = df
    ctx.embeddings = X
    return df

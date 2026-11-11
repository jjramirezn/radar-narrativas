"""Probe: which comments should Claude label? Compares selection strategies.

For each narrative, a budget of comments gets labels (from the answer key, standing in
for Claude) and a logistic regression on embeddings labels the rest. Strategies:
    random   uniform sample
    diverse  k-means on the narrative's embeddings, the comment nearest each centroid
    active   half diverse, then half where a first classifier is least sure
    adaptive half diverse; if those labels look lopsided (minority < 25%) the other
             half goes where the classifier is least sure, otherwise more diverse picks
    oracle   balanced by the true stance: impossible on Friday, upper reference only
Scenarios: "natural" (the practice mix, about 50/50) and "skewed" (rejecting comments
subsampled to 15%, like a lopsided real-world narrative).

Usage:
    python eval/probe_label_selection.py --datos desafio-qualydata/datos/practica
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

STRATEGIES = ["random", "diverse", "active", "adaptive", "oracle"]


def diverse(X, idx, k, seed):
    km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(X[idx])
    picks = []
    for c in range(k):
        members = idx[km.labels_ == c]
        picks.append(members[np.argmin(((X[members] - km.cluster_centers_[c]) ** 2).sum(1))])
    return np.array(picks)


def select(strategy, X, y, idx, k, seed, C):
    rng = np.random.default_rng(seed)
    if strategy == "random":
        return rng.choice(idx, k, replace=False)
    if strategy == "oracle":
        pos, neg = idx[y[idx]], idx[~y[idx]]
        n_neg = min(len(neg), k // 2)
        return np.concatenate([rng.choice(pos, k - n_neg, replace=False), rng.choice(neg, n_neg, replace=False)])
    if strategy == "diverse":
        return diverse(X, idx, k, seed)
    first = diverse(X, idx, k // 2, seed)
    rest = np.setdiff1d(idx, first)
    minority_share = min(y[first].mean(), 1 - y[first].mean())
    if strategy == "adaptive" and minority_share >= 0.25:
        # Balanced enough after the first half: keep covering distinct arguments.
        extra = [i for i in diverse(X, idx, k, seed) if i not in set(first)]
        extra += list(rng.choice(np.setdiff1d(rest, extra), max(0, k - len(first) - len(extra)), replace=False))
        return np.concatenate([first, np.array(extra[: k - len(first)], dtype=int)])
    if len(set(y[first])) < 2:
        second = rng.choice(rest, k - k // 2, replace=False)
    else:
        proba = LogisticRegression(C=C, max_iter=2000).fit(X[first], y[first]).predict_proba(X[rest])[:, 1]
        second = rest[np.argsort(np.abs(proba - 0.5))[: k - k // 2]]
    return np.concatenate([first, second])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=Path("desafio-qualydata/datos/practica"))
    ap.add_argument("--por-narrativa", type=int, default=1000)
    ap.add_argument("--presupuestos", type=int, nargs="+", default=[20, 60], help="Labels per narrative")
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--modelo", default="intfloat/multilingual-e5-small")
    args = ap.parse_args()
    d = args.datos

    rows = duckdb.sql(f"""
        WITH t AS (
            SELECT c.texto, v.narrativa_id, v.postura,
                   row_number() OVER (PARTITION BY v.narrativa_id ORDER BY hash(c.comentario_id)) AS rk
            FROM read_csv('{d}/comentarios.csv.gz', all_varchar = true) c
            JOIN read_csv('{d}/verdad/comentarios.csv.gz', all_varchar = true) v USING (comentario_id)
            WHERE v.narrativa_id <> 'ninguna' AND len(string_split(trim(c.texto), ' ')) > 5
        )
        SELECT texto, narrativa_id, postura FROM t WHERE rk <= {args.por_narrativa}
    """).fetchall()
    X = SentenceTransformer(args.modelo, device="cpu").encode(
        [f"query: {r[0]}" for r in rows], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    narr = np.array([r[1] for r in rows])
    y = np.array([r[2] == "apoya" for r in rows])

    rng = np.random.default_rng(42)
    scenarios = {"natural": {}, "skewed": {}}
    for n in sorted(set(narr)):
        idx = np.where(narr == n)[0]
        pos, neg = idx[y[idx]], idx[~y[idx]]
        scenarios["natural"][n] = idx
        scenarios["skewed"][n] = np.concatenate([pos, rng.choice(neg, int(0.176 * len(pos)), replace=False)])

    print("stance macro-F1 on unlabeled comments (mean over seeds); 'minority' = avg rejecting comments picked")
    for scenario, groups in scenarios.items():
        for k in args.presupuestos:
            cells = []
            for strategy in STRATEGIES:
                scores, minority = [], []
                for seed in range(args.semillas):
                    y_true, y_pred = [], []
                    for idx in groups.values():
                        train = select(strategy, X, y, idx, k, seed, args.C)
                        test = np.setdiff1d(idx, train)
                        minority.append((~y[train]).sum())
                        if len(set(y[train])) < 2:
                            pred = np.full(len(test), y[train][0])
                        else:
                            pred = LogisticRegression(C=args.C, max_iter=2000).fit(X[train], y[train]).predict(X[test])
                        y_true.append(y[test])
                        y_pred.append(pred)
                    scores.append(f1_score(np.concatenate(y_true), np.concatenate(y_pred), average="macro"))
                cells.append(f"{strategy} {np.mean(scores):.3f} (minority {np.mean(minority):4.1f})")
            print(f"{scenario:8s} k={k:3d}  " + "  |  ".join(cells))


if __name__ == "__main__":
    main()

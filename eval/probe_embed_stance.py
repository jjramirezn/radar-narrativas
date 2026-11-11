"""Probe: can a tiny per-narrative classifier on embeddings learn stance from a few labels?

For each narrative: sample comments, embed them, train a logistic regression on k
labeled examples (balanced), and test on the rest. Labels come from the answer key
here, standing in for labels Claude or the NLI model would give on Friday.

Short replies (<=5 words, like "Tal cual") are skipped: the pipeline gives them the
stance of the comment they answer.

Usage:
    python eval/probe_embed_stance.py --datos desafio-qualydata/datos/practica
"""
import argparse
import time
from pathlib import Path

import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=Path("desafio-qualydata/datos/practica"))
    ap.add_argument("--por-narrativa", type=int, default=1000, help="Comments sampled per narrative")
    ap.add_argument("--etiquetas", type=int, nargs="+", default=[20, 60, 200], help="Labeled examples per narrative")
    ap.add_argument("--C", type=float, nargs="+", default=[1.0, 10.0, 100.0], help="Inverse regularization strengths")
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

    model = SentenceTransformer(args.modelo, device="cpu")
    started = time.time()
    # e5 models expect a "query: " prefix for symmetric tasks like classification.
    emb = model.encode([f"query: {r[0]}" for r in rows], batch_size=128, normalize_embeddings=True)
    speed = len(rows) / (time.time() - started)
    narr = np.array([r[1] for r in rows])
    y = np.array([r[2] == "apoya" for r in rows])
    print(f"model: {args.modelo}  embedded {len(rows)} comments at {speed:.0f}/s on CPU")
    print("reference: keyword baseline 0.491, zero-shot NLI 0.714")

    for k in args.etiquetas:
        for C in args.C:
            rng = np.random.default_rng(0)
            y_true, y_pred, per = [], [], {}
            for n in sorted(set(narr)):
                idx = np.where(narr == n)[0]
                pos, neg = idx[y[idx]], idx[~y[idx]]
                train = np.concatenate([rng.choice(pos, k // 2, replace=False), rng.choice(neg, k // 2, replace=False)])
                test = np.setdiff1d(idx, train)
                clf = LogisticRegression(C=C, max_iter=2000).fit(emb[train], y[train])
                pred = clf.predict(emb[test])
                y_true.append(y[test])
                y_pred.append(pred)
                per[n] = round(float((pred == y[test]).mean()), 2)
            macro = f1_score(np.concatenate(y_true), np.concatenate(y_pred), average="macro")
            print(f"{k:4d} labels/narrative  C={C:<6g} stance macro-F1 = {macro:.3f}  {per}")


if __name__ == "__main__":
    main()

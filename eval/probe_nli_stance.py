"""Probe: how well does a zero-shot multilingual NLI model judge stance on the practice set?

An NLI model reads a premise and a hypothesis together and says whether the premise
entails or contradicts it. Premise = the comment, hypothesis = the narrative's claim
(its title from the answer key, standing in for the title Claude will write on Friday).
Supports if P(entailment) > P(contradiction), else rejects.

Short replies (<=5 words, like "Tal cual") are skipped: they inherit stance from the
comment they answer, which the pipeline handles with a rule.

Usage:
    python eval/probe_nli_stance.py --datos desafio-qualydata/datos/practica --por-clase 20
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=Path("desafio-qualydata/datos/practica"))
    ap.add_argument("--por-clase", type=int, default=20, help="Samples per narrative and stance")
    ap.add_argument("--modelo", default=DEFAULT_MODEL)
    args = ap.parse_args()
    d = args.datos

    rows = duckdb.sql(f"""
        WITH t AS (
            SELECT c.texto, v.narrativa_id, v.postura, n.titulo,
                   row_number() OVER (PARTITION BY v.narrativa_id, v.postura ORDER BY hash(c.comentario_id)) AS rk
            FROM read_csv('{d}/comentarios.csv.gz', all_varchar = true) c
            JOIN read_csv('{d}/verdad/comentarios.csv.gz', all_varchar = true) v USING (comentario_id)
            JOIN read_csv('{d}/verdad/narrativas.csv', all_varchar = true) n USING (narrativa_id)
            WHERE v.narrativa_id <> 'ninguna' AND len(string_split(trim(c.texto), ' ')) > 5
        )
        SELECT texto, narrativa_id, postura, titulo FROM t WHERE rk <= {args.por_clase}
        ORDER BY narrativa_id, postura
    """).fetchall()

    started = time.time()
    tok = AutoTokenizer.from_pretrained(args.modelo)
    model = AutoModelForSequenceClassification.from_pretrained(args.modelo).eval()
    loaded = time.time()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    ent, contra = labels["entailment"], labels["contradiction"]

    preds = []
    with torch.inference_mode():
        for i in range(0, len(rows), 32):
            batch = rows[i:i + 32]
            enc = tok([r[0] for r in batch], [r[3] for r in batch], truncation=True, padding=True, return_tensors="pt")
            logits = model(**enc).logits
            preds += ["apoya" if row[ent] > row[contra] else "rechaza" for row in logits]
    elapsed = time.time() - loaded

    stats = {cls: [0, 0, 0] for cls in ("apoya", "rechaza")}
    per_narr = defaultdict(lambda: [0, 0])
    errors = []
    for (texto, narr, truth, titulo), pred in zip(rows, preds):
        for cls in stats:
            stats[cls][0] += truth == cls and pred == cls
            stats[cls][1] += truth != cls and pred == cls
            stats[cls][2] += truth == cls and pred != cls
        per_narr[narr][0] += truth == pred
        per_narr[narr][1] += 1
        if truth != pred and len(errors) < 8:
            errors.append(f"  [{narr} truth={truth}] {texto[:100]!r}")

    macro = sum(f1(*s) for s in stats.values()) / 2
    print(f"model: {args.modelo} (loaded in {loaded - started:.0f}s)")
    print(f"pairs: {len(rows)}  speed: {len(rows) / elapsed:.1f} pairs/s on CPU")
    print(f"stance macro-F1: {macro:.3f}   (baseline keyword list: 0.491)")
    print("accuracy by narrative:", {n: round(c / t, 2) for n, (c, t) in sorted(per_narr.items())})
    print("sample errors:")
    print("\n".join(errors))


if __name__ == "__main__":
    main()

"""Score the full stance step inside the pipeline, with three labelers:
    oracle  the answer key stands in for Claude (perfect labels: the ceiling of the method)
    noisy   the answer key with a share of labels flipped (how Claude's mistakes would hurt)
    none    the no-AI fallback (reply-direction split)

Usage:
    python eval/probe_stance_pipeline.py --datos desafio-qualydata/datos/practica
"""
import argparse
import importlib.util
import tempfile
from pathlib import Path

import duckdb
import numpy as np

from radar import ingest, narratives, output, stance
from radar.context import Context, Settings
from radar.llm import LLM


def load_scorer(path: Path):
    spec = importlib.util.spec_from_file_location("puntuar", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def truth_labeler(datos: Path, flip: float, seed: int = 0):
    truth = dict(duckdb.sql(f"""
        SELECT c.texto, mode(v.postura) FROM read_csv('{datos}/comentarios.csv.gz', all_varchar = true) c
        JOIN read_csv('{datos}/verdad/comentarios.csv.gz', all_varchar = true) v USING (comentario_id)
        WHERE v.narrativa_id <> 'ninguna' GROUP BY c.texto
    """).fetchall())
    rng = np.random.default_rng(seed)
    swap = {"apoya": "rechaza", "rechaza": "apoya"}

    def label(jobs):
        out = []
        for _claim, texts in jobs:
            labels = [truth.get(t, "otra") for t in texts]
            out.append([swap.get(lab, lab) if rng.random() < flip else lab for lab in labels])
        return out
    return label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=Path("desafio-qualydata/datos/practica"))
    ap.add_argument("--scorer", type=Path, default=Path("desafio-qualydata/evaluacion/puntuar.py"))
    ap.add_argument("--flips", type=float, nargs="+", default=[0.1, 0.2])
    args = ap.parse_args()
    scorer = load_scorer(args.scorer)

    with tempfile.TemporaryDirectory() as tmp:
        ctx = Context(datos=args.datos, entrega=Path(tmp) / "entrega", trabajo=Path("trabajo") / args.datos.resolve().name,
                      settings=Settings(use_llm=False))
        ingest.run(ctx)
        narratives.run(ctx)
        claims = dict(ctx.con.execute("SELECT narrativa, titulo FROM narrativas").fetchall())
        llm = LLM(ctx)
        runs = [("oracle", truth_labeler(args.datos, 0.0))]
        runs += [(f"noisy {int(f * 100)}%", truth_labeler(args.datos, f)) for f in args.flips]
        runs += [("none (no AI)", None)]
        base = ctx.comments.copy()
        for name, labeler in runs:
            ctx.comments = base.copy()
            if labeler is None:
                df = stance.run(ctx, llm, claims)
            else:
                df = stance.run(ctx, llm, claims, labeler=labeler)
            ctx.put_table("etiquetas", df[["comentario_id", "narrativa", "postura"]])
            output.write(ctx)
            s = scorer.score_narratives(args.datos / "verdad", ctx.entrega)
            print(f"{name:14s} stance macro-F1 = {s['postura_macro_f1']:.3f}   narratives points = {s['puntos']}", flush=True)


if __name__ == "__main__":
    main()

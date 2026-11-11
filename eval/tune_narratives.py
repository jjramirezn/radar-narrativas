"""Grid-search narrative settings against an answer key, reusing cached embeddings.

Usage:
    python eval/tune_narratives.py --datos desafio-qualydata/datos/practica
"""
import argparse
import importlib.util
import itertools
import tempfile
from pathlib import Path

from radar import ingest, narratives, output
from radar.context import Context, Settings


def load_scorer(path: Path):
    spec = importlib.util.spec_from_file_location("puntuar", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=Path("desafio-qualydata/datos/practica"))
    ap.add_argument("--verdad", type=Path, default=None)
    ap.add_argument("--scorer", type=Path, default=Path("desafio-qualydata/evaluacion/puntuar.py"))
    ap.add_argument("--sim", type=float, nargs="+", default=[0.92])
    ap.add_argument("--res", type=float, nargs="+", default=[1.0])
    ap.add_argument("--k-div", type=float, nargs="+", default=[300.0], help="Units per micro-cluster")
    ap.add_argument("--method", nargs="+", default=["lift"])
    ap.add_argument("--noise-ratio", type=float, nargs="+", default=[0.3])
    args = ap.parse_args()
    verdad = args.verdad or args.datos / "verdad"
    scorer = load_scorer(args.scorer)

    with tempfile.TemporaryDirectory() as tmp:
        ctx = Context(datos=args.datos, entrega=Path(tmp) / "entrega", trabajo=Path("trabajo") / args.datos.resolve().name,
                      settings=Settings(use_llm=False))
        ingest.run(ctx)
        for sim, res, kdiv, nr, method in itertools.product(args.sim, args.res, args.k_div, args.noise_ratio, args.method):
            narratives.UNITS_PER_MICRO_CLUSTER = kdiv
            narratives.NOISE_CONNECTED_RATIO = nr
            df = narratives.run(ctx, sim_floor=sim, resolution=res, method=method)
            ctx.put_table("etiquetas", df[["comentario_id", "narrativa"]].assign(postura="apoya"))
            output.write(ctx)
            s = scorer.score_narratives(verdad, ctx.entrega)
            print(f"{method:16s} sim={sim:<5} res={res:<4} units/mc={kdiv:<5g} noise_ratio={nr:<4} -> "
                  f"B3 P={s['bcubed_precision']:.3f} R={s['bcubed_recall']:.3f} F1={s['bcubed_f1']:.3f}  "
                  f"noise F1={s['ruido_f1']:.3f}  origin={s['origen']:.2f}  narratives={df['narrativa'].nunique() - 1}",
                  flush=True)


if __name__ == "__main__":
    main()

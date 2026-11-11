"""Which signals separate campaigns from genuine groups? Joins candidate communities with the answer key.

Usage:
    python eval/explore_coordination.py --datos desafio-qualydata/datos/practica
"""
import argparse
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from radar import coordination, ingest, narratives
from radar.context import Context, Settings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=Path("desafio-qualydata/datos/practica"))
    args = ap.parse_args()
    ctx = Context(datos=args.datos, entrega=Path("salida/explore"), trabajo=Path("trabajo") / args.datos.resolve().name,
                  settings=Settings(use_llm=False))
    ingest.run(ctx)
    narratives.run(ctx)
    df = ctx.comments

    t0 = time.time()
    co, _ = coordination.coactivity_edges(df)
    t1 = time.time()
    rare = coordination.rare_token_edges(df)
    rings = coordination.reply_edges(df)
    print(f"edges: coactivity {len(co)} ({t1 - t0:.1f}s), rare text/hashtag {len(rare)}, reply rings {len(rings)}")

    truth = dict(duckdb.sql(f"SELECT cuenta_id, coalesce(campana_id, '') FROM "
                            f"read_csv('{args.datos}/verdad/cuentas.csv.gz', all_varchar = true)").fetchall())
    for name, edges in (("coactivity", co), ("rare", rare), ("rings", rings)):
        kinds = pd.Series([("camp" if truth.get(a) and truth.get(a) == truth.get(b) else
                            "mixed" if truth.get(a) or truth.get(b) else "genuine") for a, b in edges]).value_counts()
        print(f"  {name:10s} edge types: {dict(kinds)}")

    stats = coordination.run(ctx)
    groups = ctx.coord_groups.copy()
    groups["camp"] = groups["cuenta_id"].map(truth).fillna("")
    acc = ctx.con.execute("SELECT cuenta_id, fecha_creacion, seguidores FROM cuentas").df()
    start = pd.to_datetime(df["t"].min(), unit="s")
    acc["edad_dias"] = (start - pd.to_datetime(acc["fecha_creacion"])).dt.days
    per_acc = df.groupby("cuenta_id").agg(n=("t", "size"), hora_std=("t", lambda s: pd.to_datetime(s, unit="s").dt.hour.std()))
    focus = (df[df["narrativa"] != ""].groupby("cuenta_id")["narrativa"]
             .agg(lambda s: s.value_counts(normalize=True).iloc[0]).rename("foco"))
    groups = groups.merge(acc, on="cuenta_id", how="left").merge(per_acc, on="cuenta_id", how="left") \
        .merge(focus, on="cuenta_id", how="left")
    summary = groups.groupby("comunidad").agg(
        cuentas=("cuenta_id", "size"), campana=("camp", lambda s: s.value_counts().index[0]),
        pureza=("camp", lambda s: s.value_counts(normalize=True).iloc[0]),
        edad_med=("edad_dias", "median"), coment_med=("n", "median"), foco_med=("foco", "median")
    ).merge(stats, on="comunidad").sort_values("peso_medio", ascending=False)
    pd.set_option("display.width", 200)
    print(summary.head(40).to_string(index=False))
    print(f"campaign accounts in any community: {groups[groups['camp'] != ''].shape[0]} of {sum(1 for v in truth.values() if v)}")


if __name__ == "__main__":
    main()

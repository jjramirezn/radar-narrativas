"""Explain a run's narrative score: how predicted narratives map onto the answer key.

Prints, for each predicted narrative, which true narratives it holds; for each true
narrative, how it was split; noise confusion; and origin errors for emergent narratives.

Usage:
    python eval/diagnose.py --verdad desafio-qualydata/datos/practica/verdad --entrega salida/radar
"""
import argparse
from pathlib import Path

import duckdb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdad", type=Path, required=True)
    ap.add_argument("--entrega", type=Path, required=True)
    args = ap.parse_args()
    con = duckdb.connect()
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_csv('{args.verdad}/comentarios.csv.gz', all_varchar = true)")
    con.execute(f"CREATE VIEW p AS SELECT comentario_id, coalesce(narrativa, '') AS pred, coalesce(postura, '') AS pp "
                f"FROM read_csv('{args.entrega}/comentarios.csv', all_varchar = true)")
    con.execute("CREATE TABLE j AS SELECT t.narrativa_id AS truth, t.postura AS ts, "
                "CASE WHEN p.pred IN ('', 'ninguna') THEN '-' ELSE p.pred END AS pred, p.pp "
                "FROM t LEFT JOIN p USING (comentario_id)")

    print("== predicted narratives (size, top true narratives)")
    for pred, size, top in con.execute("""
        WITH c AS (SELECT pred, truth, count(*) AS n FROM j GROUP BY ALL),
        r AS (SELECT pred, truth, n, sum(n) OVER (PARTITION BY pred) AS size,
                     row_number() OVER (PARTITION BY pred ORDER BY n DESC) AS rk FROM c)
        SELECT pred, any_value(size), string_agg(truth || ' ' || round(100.0 * n / size)::INT || '%', ', ' ORDER BY rk)
        FROM r WHERE rk <= 3 GROUP BY pred ORDER BY any_value(size) DESC
    """).fetchall():
        print(f"  {pred:5s} {size:7d}  {top}")

    print("== true narratives (size, how they were split)")
    for truth, size, top in con.execute("""
        WITH c AS (SELECT truth, pred, count(*) AS n FROM j GROUP BY ALL),
        r AS (SELECT truth, pred, n, sum(n) OVER (PARTITION BY truth) AS size,
                     row_number() OVER (PARTITION BY truth ORDER BY n DESC) AS rk FROM c)
        SELECT truth, any_value(size), string_agg(pred || ' ' || round(100.0 * n / size)::INT || '%', ', ' ORDER BY rk)
        FROM r WHERE rk <= 4 GROUP BY truth ORDER BY truth
    """).fetchall():
        print(f"  {truth:8s} {size:7d}  {top}")

    tp, fp, fn = con.execute("""
        SELECT count(*) FILTER (truth = 'ninguna' AND pred = '-'), count(*) FILTER (truth <> 'ninguna' AND pred = '-'),
               count(*) FILTER (truth = 'ninguna' AND pred <> '-') FROM j
    """).fetchone()
    print(f"== noise: caught {tp}, wrongly called noise {fp}, missed {fn}")

    narr = args.verdad / "narrativas.csv"
    sub = args.entrega / "narrativas.csv"
    if narr.exists() and sub.exists():
        print("== emergent narrative origins (hours off; full credit within 3h)")
        for row in con.execute(f"""
            WITH counts AS (SELECT truth, pred, count(*) AS n FROM j WHERE pred <> '-' GROUP BY ALL),
            best AS (
                SELECT truth, pred FROM (SELECT truth, pred,
                       row_number() OVER (PARTITION BY truth ORDER BY n DESC) AS rk FROM counts) WHERE rk = 1)
            SELECT v.narrativa_id, b.pred, v.fecha_origen, s.fecha_origen,
                   round((epoch(s.fecha_origen::TIMESTAMPTZ) - epoch(v.fecha_origen::TIMESTAMPTZ)) / 3600, 1)
            FROM read_csv('{narr}', all_varchar = true) v
            LEFT JOIN best b ON b.truth = v.narrativa_id
            LEFT JOIN read_csv('{sub}', all_varchar = true) s ON s.narrativa = b.pred
            WHERE v.emergente = 'si'
        """).fetchall():
            print(f"  {row[0]} -> {row[1]}: true {row[2]}  ours {row[3]}  ({row[4]:+}h)" if row[4] is not None
                  else f"  {row[0]} -> {row[1]}: no origin")


if __name__ == "__main__":
    main()

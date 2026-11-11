"""Split the practice set into single-topic slices, each with its own answer key.

Every practice post is about exactly one topic, so slicing by post is clean. The
topic comes from the answer key (narrative id prefix: L=litio, B=colectivo,
H=hospital). That is fine here because this is evaluation tooling: slices are
only used to check that settings hold up on topics they weren't tuned on.

Usage:
    python eval/slices.py --datos desafio-qualydata/datos/practica --salida salida/slices
"""
import argparse
import shutil
from pathlib import Path

import duckdb

TOPICS = {"L": "litio", "B": "colectivo", "H": "hospital"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=Path("desafio-qualydata/datos/practica"))
    ap.add_argument("--salida", type=Path, default=Path("salida/slices"))
    args = ap.parse_args()
    d, v = args.datos, args.datos / "verdad"

    con = duckdb.connect()
    for name, path in [("c", d / "comentarios.csv.gz"), ("a", d / "cuentas.csv.gz"),
                       ("p", d / "publicaciones.csv.gz"), ("tc", v / "comentarios.csv.gz"),
                       ("ta", v / "cuentas.csv.gz"), ("tn", v / "narrativas.csv"), ("tk", v / "campanas.csv")]:
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv(?, all_varchar = true)", [str(path)])
    con.execute("""
        CREATE TABLE post_topic AS
        SELECT c.publicacion_id, mode(left(tc.narrativa_id, 1)) AS tema
        FROM c JOIN tc USING (comentario_id)
        WHERE tc.narrativa_id <> 'ninguna'
        GROUP BY 1
    """)

    for prefix, topic in TOPICS.items():
        out = args.salida / topic
        (out / "verdad").mkdir(parents=True, exist_ok=True)
        # Views can't take prepared parameters; prefix is one of our own TOPICS keys.
        con.execute(f"CREATE OR REPLACE TEMP VIEW sc AS "
                    f"SELECT c.* FROM c JOIN post_topic USING (publicacion_id) WHERE tema = '{prefix}'")
        # Campaign accounts also post cover comments on other topics. In a slice they only
        # count as coordinated if their campaign pushes a narrative of that slice's topic.
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW sta AS
            SELECT ta.cuenta_id,
                   CASE WHEN left(tk.narrativa_id, 1) = '{prefix}' THEN ta.campana_id END AS campana_id
            FROM ta LEFT JOIN tk USING (campana_id)
            WHERE ta.cuenta_id IN (SELECT cuenta_id FROM sc)
        """)
        queries = {
            out / "comentarios.csv.gz": "SELECT * FROM sc ORDER BY comentario_id",
            out / "publicaciones.csv.gz":
                "SELECT * FROM p WHERE publicacion_id IN (SELECT publicacion_id FROM sc) ORDER BY publicacion_id",
            out / "cuentas.csv.gz": "SELECT * FROM a WHERE cuenta_id IN (SELECT cuenta_id FROM sc) ORDER BY cuenta_id",
            out / "verdad/comentarios.csv.gz":
                "SELECT * FROM tc WHERE comentario_id IN (SELECT comentario_id FROM sc) ORDER BY comentario_id",
            out / "verdad/cuentas.csv.gz": "SELECT * FROM sta ORDER BY cuenta_id",
            out / "verdad/narrativas.csv": f"SELECT * FROM tn WHERE left(narrativa_id, 1) = '{prefix}'",
            out / "verdad/campanas.csv": f"SELECT * FROM tk WHERE left(narrativa_id, 1) = '{prefix}'",
        }
        for path, query in queries.items():
            con.execute(f"COPY ({query}) TO '{path}' (HEADER)")
        shutil.copy(d / "fuentes.csv", out / "fuentes.csv")

        n_com, n_acc, n_camp = con.execute("""
            SELECT (SELECT count(*) FROM sc),
                   (SELECT count(DISTINCT cuenta_id) FROM sc),
                   (SELECT count(*) FROM sta WHERE campana_id IS NOT NULL)
        """).fetchone()
        print(f"{topic:10s} {n_com:7d} comments  {n_acc:6d} accounts  {n_camp:4d} campaign accounts -> {out}")


if __name__ == "__main__":
    main()

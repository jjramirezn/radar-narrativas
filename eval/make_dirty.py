"""Make a deliberately dirty copy of a dataset, to test that the pipeline survives messy input.

Corruptions (all seeded, so the copy is reproducible):
  comentarios: duplicated rows, empty texts, malformed / mixed-format / timezone-less dates,
               replies to comments that don't exist, non-numeric likes, line breaks and quotes
               inside texts, a broken CSV line, an extra column, shuffled column order, a BOM
  cuentas:     duplicated ids, dates in another format, non-numeric follower counts, missing accounts
  publicaciones: missing posts
  fuentes:     file removed
The answer key is copied as is, so scores stay comparable with the clean run.

Usage:
    python eval/make_dirty.py --datos desafio-qualydata/datos/practica --salida salida/sucio
"""
import argparse
import csv
import gzip
import random
import shutil
from pathlib import Path


def read(path: Path) -> tuple[list[str], list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write(path: Path, header: list[str], rows: list[dict], bom: bool = False, broken_line: str | None = None):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        if bom:
            f.write("﻿")
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(rows):
            w.writerow(row)
            if broken_line and i == len(rows) // 2:
                f.write(broken_line + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=Path("desafio-qualydata/datos/practica"))
    ap.add_argument("--salida", type=Path, default=Path("salida/sucio"))
    ap.add_argument("--semilla", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.semilla)
    out = args.salida
    out.mkdir(parents=True, exist_ok=True)
    stats = {}

    header, rows = read(args.datos / "comentarios.csv.gz")
    for r in rng.sample(rows, len(rows) // 200):
        r["texto"] = ""
    for r in rng.sample(rows, len(rows) // 200):
        r["fecha"] = rng.choice(["2026-13-45T99:00:00", "ayer", "", "N/A"])
    for r in rng.sample(rows, len(rows) // 100):
        d = r["fecha"]
        r["fecha"] = rng.choice([d[:19].replace("T", " "), f"{d[8:10]}/{d[5:7]}/{d[:4]} {d[11:19]}"])
    for r in rng.sample(rows, len(rows) // 200):
        r["respuesta_a"] = f"c9{rng.randint(0, 9999999):07d}"
    for r in rng.sample(rows, len(rows) // 200):
        r["me_gusta"] = rng.choice(["12k", "", "-3", "muchos"])
    for r in rng.sample(rows, len(rows) // 200):
        r["texto"] = r["texto"].replace(" ", "\n", 1) + ' "citado" \x07'
    dupes = [dict(r) for r in rng.sample(rows, len(rows) // 100)]
    rows += dupes
    rng.shuffle(rows)
    for r in rows:
        r["origen_app"] = rng.choice(["android", "ios", "web"])
    new_header = ["texto", "comentario_id", "fecha", "cuenta_id", "origen_app", "publicacion_id", "me_gusta", "respuesta_a"]
    write(out / "comentarios.csv.gz", new_header, rows, bom=True,
          broken_line='c_roto,"texto sin cerrar,2026-09-01,u1')
    stats["comentarios"] = f"{len(rows)} rows ({len(dupes)} duplicated)"

    header, accs = read(args.datos / "cuentas.csv.gz")
    dropped = set(r["cuenta_id"] for r in rng.sample(accs, len(accs) // 100))
    accs = [r for r in accs if r["cuenta_id"] not in dropped]
    for r in rng.sample(accs, len(accs) // 50):
        y, m, d = r["fecha_creacion"].split("-")
        r["fecha_creacion"] = f"{d}/{m}/{y}"
    for r in rng.sample(accs, len(accs) // 100):
        r["seguidores"] = rng.choice(["1.2k", "", "?"])
    accs += [dict(r) for r in rng.sample(accs, len(accs) // 100)]
    write(out / "cuentas.csv.gz", header, accs)
    stats["cuentas"] = f"{len(accs)} rows, {len(dropped)} accounts removed"

    header, posts = read(args.datos / "publicaciones.csv.gz")
    posts = posts[5:]
    write(out / "publicaciones.csv.gz", header, posts)
    stats["publicaciones"] = "first 5 posts removed"

    if (args.datos / "verdad").exists():
        shutil.copytree(args.datos / "verdad", out / "verdad", dirs_exist_ok=True)
    print(f"dirty copy in {out}: {stats}; fuentes.csv left out")


if __name__ == "__main__":
    main()

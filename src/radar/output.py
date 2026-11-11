"""Write the four deliverable files into the entrega folder.

Pipeline steps leave their results in three DuckDB tables, all optional:
    etiquetas(comentario_id, narrativa, postura)
    narrativas(narrativa, titulo, frases_clave, origen_epoch)
    grupos(cuenta_id, grupo)
Every input comment and account gets a row, in input order, even if a step failed.
"""
import json
from datetime import datetime, timedelta, timezone

from .context import Context


def table_exists(ctx: Context, name: str) -> bool:
    return bool(ctx.sql("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name])[0][0])


def iso(epoch: float, offset: str) -> str:
    sign = -1 if offset.startswith("-") else 1
    hours, minutes = int(offset[1:3]), int(offset[4:6])
    tz = timezone(sign * timedelta(hours=hours, minutes=minutes))
    return datetime.fromtimestamp(epoch, tz=tz).isoformat(timespec="seconds")


def write(ctx: Context):
    con, out = ctx.con, ctx.entrega
    if not table_exists(ctx, "etiquetas"):
        ctx.sql("CREATE TABLE etiquetas (comentario_id VARCHAR, narrativa VARCHAR, postura VARCHAR)")
    if not table_exists(ctx, "narrativas"):
        ctx.sql("CREATE TABLE narrativas (narrativa VARCHAR, titulo VARCHAR, frases_clave VARCHAR, origen_epoch DOUBLE)")
    if not table_exists(ctx, "grupos"):
        ctx.sql("CREATE TABLE grupos (cuenta_id VARCHAR, grupo VARCHAR)")

    ctx.sql(f"""
        COPY (
            SELECT c.comentario_id, coalesce(e.narrativa, '') AS narrativa,
                   CASE WHEN coalesce(e.narrativa, '') = '' THEN '' ELSE coalesce(e.postura, '') END AS postura
            FROM comentarios c LEFT JOIN etiquetas e USING (comentario_id)
            ORDER BY c.orden
        ) TO '{out / "comentarios.csv"}' (HEADER)
    """)

    offset = ctx.report.get("ingesta", {}).get("utc_offset", "-03:00")
    rows = ctx.sql("SELECT narrativa, titulo, frases_clave, origen_epoch FROM narrativas ORDER BY narrativa")
    ctx.sql("CREATE OR REPLACE TEMP TABLE narrativas_salida (narrativa VARCHAR, titulo VARCHAR, "
            "frases_clave VARCHAR, fecha_origen VARCHAR)")
    if rows:
        con.executemany("INSERT INTO narrativas_salida VALUES (?, ?, ?, ?)",
                        [(n, t, f, iso(o, offset) if o is not None else "") for n, t, f, o in rows])
    ctx.sql(f"COPY narrativas_salida TO '{out / 'narrativas.csv'}' (HEADER)")

    ctx.sql(f"""
        COPY (
            SELECT a.cuenta_id, coalesce(g.grupo, '') AS grupo
            FROM cuentas a LEFT JOIN grupos g USING (cuenta_id)
            ORDER BY a.cuenta_id
        ) TO '{out / "cuentas.csv"}' (HEADER)
    """)

    comments = ctx.sql("SELECT count(*) FROM comentarios")[0][0]
    cost = ctx.costs.to_json(comments, ctx.settings)
    (out / "costo.json").write_text(json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8")
    return cost

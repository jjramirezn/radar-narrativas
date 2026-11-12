"""Load the challenge files into clean, typed Parquet tables using DuckDB.

The final dataset may be dirtier than the practice one, so nothing here trusts the
input: files may be gzipped or plain, columns may be missing, extra or reordered,
ids may repeat, dates may be malformed, replies may point at comments that don't
exist. Everything is read as text, then cleaned and typed explicitly. Every problem
is counted in the run report instead of stopping the run.
"""
from pathlib import Path

from .context import Context

SCHEMAS = {
    "comentarios": ["comentario_id", "publicacion_id", "cuenta_id", "fecha", "texto", "respuesta_a", "me_gusta"],
    "cuentas": ["cuenta_id", "usuario", "fecha_creacion", "seguidores", "siguiendo"],
    "publicaciones": ["publicacion_id", "fuente_id", "fecha", "titular"],
    "fuentes": ["fuente_id", "nombre", "tipo"],
}

# Tried in order when a date isn't ISO 8601 (which DuckDB casts directly).
DATE_FORMATS = ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d",
                "%d-%m-%Y %H:%M:%S", "%d-%m-%Y"]


def find_input(datos: Path, name: str) -> Path | None:
    for suffix in (".csv.gz", ".csv", ".parquet"):
        path = datos / f"{name}{suffix}"
        if path.exists():
            return path
    return None


def parse_ts(col: str) -> str:
    """SQL expression turning a text column into TIMESTAMPTZ: ISO, common formats, or epoch."""
    formats = ", ".join(f"'{f}'" for f in DATE_FORMATS)
    return f"""coalesce(
        TRY_CAST({col} AS TIMESTAMPTZ),
        try_strptime({col}, [{formats}])::TIMESTAMPTZ,
        CASE WHEN TRY_CAST({col} AS DOUBLE) BETWEEN 1e9 AND 1e10 THEN to_timestamp(TRY_CAST({col} AS DOUBLE))
             WHEN TRY_CAST({col} AS DOUBLE) BETWEEN 1e12 AND 1e13 THEN to_timestamp(TRY_CAST({col} AS DOUBLE) / 1000)
        END)"""


def load_raw(ctx: Context, name: str) -> dict:
    """Create raw_<name> with exactly the expected columns, all VARCHAR, plus file order."""
    cols = SCHEMAS[name]
    path = find_input(ctx.datos, name)
    info = {"archivo": path.name if path else None}
    rejects_before = _rejected(ctx)
    if path is None:
        if name == "comentarios":
            raise SystemExit(f"Falta comentarios.csv(.gz) en {ctx.datos}")
        empty = ", ".join(f"NULL::VARCHAR AS {c}" for c in cols)
        ctx.con.execute(f"CREATE OR REPLACE TABLE raw_{name} AS SELECT {empty}, 0::BIGINT AS orden WHERE false")
        return info

    if path.suffix == ".parquet":
        sources = [f"read_parquet('{path}')"]
    else:
        # Strictest first. store_rejects skips and counts unparseable lines; the lenient last
        # resort can silently drop many rows around a broken quote, so it's only a fallback.
        opts = "all_varchar = true, header = true"
        sources = [f"read_csv('{path}', {opts}, store_rejects = true)",
                   f"read_csv('{path}', {opts}, store_rejects = true, encoding = 'latin-1')",
                   f"read_csv('{path}', {opts}, ignore_errors = true, null_padding = true)"]
    for i, source in enumerate(sources):
        try:
            ctx.con.execute(f"CREATE OR REPLACE TEMP VIEW src_{name} AS SELECT * FROM {source}")
            present = {}
            for col in ctx.con.execute(f"DESCRIBE src_{name}").fetchall():
                present.setdefault(col[0].strip().lstrip("﻿").lower(), col[0])
            select = ", ".join(f'CAST("{present[c]}" AS VARCHAR) AS {c}' if c in present else f"NULL::VARCHAR AS {c}"
                               for c in cols)
            ctx.con.execute(f"CREATE OR REPLACE TABLE raw_{name} AS "
                            f"SELECT {select}, row_number() OVER () AS orden FROM src_{name}")
            info["modo_lectura"] = ["estricto", "latin-1", "tolerante"][i] if len(sources) > 1 else "parquet"
            break
        except Exception as exc:  # noqa: BLE001 - try the next way of reading the file
            if i == len(sources) - 1:
                raise
            info.setdefault("reintentos", []).append(str(exc).splitlines()[0][:160])
    info["filas"] = ctx.con.execute(f"SELECT count(*) FROM raw_{name}").fetchone()[0]
    info["columnas_faltantes"] = [c for c in cols if c not in present]
    info["lineas_rechazadas"] = _rejected(ctx) - rejects_before
    return info


def _rejected(ctx: Context) -> int:
    """Lines DuckDB has rejected so far on this connection (its reject table accumulates)."""
    try:
        return ctx.con.execute("SELECT count(*) FROM reject_errors").fetchone()[0]
    except Exception:  # noqa: BLE001 - no CSV read yet, so no reject table
        return 0


def run(ctx: Context):
    con = ctx.con
    report = {name: load_raw(ctx, name) for name in SCHEMAS}

    con.execute(f"""
        CREATE OR REPLACE TABLE comentarios AS
        WITH r AS (
            SELECT nullif(trim(comentario_id), '') AS comentario_id,
                   nullif(trim(publicacion_id), '') AS publicacion_id,
                   nullif(trim(cuenta_id), '') AS cuenta_id,
                   trim(fecha) AS fecha_txt,
                   -- control characters and line breaks become spaces
                   trim(regexp_replace(coalesce(texto, ''), '[\\x00-\\x1F\\x7F]+', ' ', 'g')) AS texto,
                   nullif(trim(respuesta_a), '') AS respuesta_a,
                   me_gusta, orden
            FROM raw_comentarios
        )
        SELECT comentario_id,
               coalesce(publicacion_id, '__sin_publicacion') AS publicacion_id,
               coalesce(cuenta_id, '__sin_cuenta') AS cuenta_id,
               {parse_ts('fecha_txt')} AS ts,
               fecha_txt,
               texto,
               respuesta_a,
               greatest(coalesce(TRY_CAST(TRY_CAST(me_gusta AS DOUBLE) AS BIGINT), 0), 0) AS me_gusta,
               orden
        FROM r
        WHERE comentario_id IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY comentario_id ORDER BY orden) = 1
    """)
    rows = report["comentarios"]["filas"]
    kept = con.execute("SELECT count(*) FROM comentarios").fetchone()[0]
    report["comentarios"]["descartados_sin_id_o_duplicados"] = rows - kept

    # Replies to unknown comments or to themselves can't be followed: drop the link.
    report["comentarios"]["respuestas_huerfanas"] = con.execute("""
        SELECT count(*) FROM comentarios c
        WHERE respuesta_a IS NOT NULL
          AND (respuesta_a = comentario_id OR NOT EXISTS (SELECT 1 FROM comentarios p WHERE p.comentario_id = c.respuesta_a))
    """).fetchone()[0]
    con.execute("""
        UPDATE comentarios c SET respuesta_a = NULL
        WHERE respuesta_a IS NOT NULL
          AND (respuesta_a = comentario_id OR NOT EXISTS (SELECT 1 FROM comentarios p WHERE p.comentario_id = c.respuesta_a))
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE publicaciones AS
        WITH d AS (
            SELECT nullif(trim(publicacion_id), '') AS publicacion_id, nullif(trim(fuente_id), '') AS fuente_id,
                   {parse_ts('trim(fecha)')} AS ts, coalesce(titular, '') AS titular
            FROM raw_publicaciones
            WHERE nullif(trim(publicacion_id), '') IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY trim(publicacion_id) ORDER BY orden) = 1
        )
        SELECT * FROM d
        UNION ALL
        SELECT publicacion_id, NULL, min(ts), '' FROM comentarios c
        WHERE NOT EXISTS (SELECT 1 FROM d WHERE d.publicacion_id = c.publicacion_id)
        GROUP BY publicacion_id
    """)
    report["publicaciones"]["creadas_por_faltantes"] = con.execute(
        "SELECT count(*) FROM publicaciones WHERE fuente_id IS NULL").fetchone()[0]

    # Missing comment dates: use the post's date, else the median comment date.
    report["comentarios"]["fechas_invalidas"] = con.execute(
        "SELECT count(*) FROM comentarios WHERE ts IS NULL").fetchone()[0]
    con.execute("""
        UPDATE comentarios c SET ts = coalesce(
            (SELECT p.ts FROM publicaciones p WHERE p.publicacion_id = c.publicacion_id),
            (SELECT to_timestamp(median(epoch(ts))) FROM comentarios WHERE ts IS NOT NULL))
        WHERE ts IS NULL
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE cuentas AS
        WITH d AS (
            SELECT nullif(trim(cuenta_id), '') AS cuenta_id, usuario,
                   coalesce(TRY_CAST(trim(fecha_creacion) AS DATE), CAST({parse_ts('trim(fecha_creacion)')} AS DATE))
                       AS fecha_creacion,
                   TRY_CAST(TRY_CAST(seguidores AS DOUBLE) AS BIGINT) AS seguidores,
                   TRY_CAST(TRY_CAST(siguiendo AS DOUBLE) AS BIGINT) AS siguiendo
            FROM raw_cuentas
            WHERE nullif(trim(cuenta_id), '') IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY trim(cuenta_id) ORDER BY orden) = 1
        )
        SELECT * FROM d
        UNION ALL
        SELECT DISTINCT cuenta_id, NULL, NULL::DATE, NULL::BIGINT, NULL::BIGINT FROM comentarios c
        WHERE NOT EXISTS (SELECT 1 FROM d WHERE d.cuenta_id = c.cuenta_id)
    """)
    report["cuentas"]["creadas_por_faltantes"] = con.execute(
        "SELECT count(*) FROM cuentas WHERE usuario IS NULL AND fecha_creacion IS NULL").fetchone()[0]

    con.execute("""
        CREATE OR REPLACE TABLE fuentes AS
        SELECT nullif(trim(fuente_id), '') AS fuente_id, nombre, tipo FROM raw_fuentes
        WHERE nullif(trim(fuente_id), '') IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY trim(fuente_id) ORDER BY orden) = 1
    """)

    # The dominant UTC offset of the input, reused when writing dates back out.
    offset = con.execute("""
        SELECT mode(regexp_extract(fecha_txt, '([+-][0-9]{2}:[0-9]{2})$', 1)) FROM comentarios
        WHERE regexp_matches(fecha_txt, '[+-][0-9]{2}:[0-9]{2}$')
    """).fetchone()[0]
    report["utc_offset"] = offset or "-03:00"

    for table in ("comentarios", "cuentas", "publicaciones", "fuentes"):
        con.execute(f"COPY {table} TO '{ctx.parquet(table)}' (FORMAT parquet, COMPRESSION zstd)")
        con.execute(f"DROP TABLE IF EXISTS raw_{table}")
    ctx.report["ingesta"] = report
    return report

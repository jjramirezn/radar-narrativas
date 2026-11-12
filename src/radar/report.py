"""Self-contained HTML dashboard (entrega/tablero.html): what circulates, since when, and who pushes it.

No external resources: inline SVG and a few lines of vanilla JS for tooltips, so it opens
offline. Light and dark themes; every number shown in a chart is also in the table view.
"""
import html
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .context import Context

BUCKET_H = 12
COLORS = """
.viz{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--total:#898781;--coord:#eb6834;--apoya:#2a78d6;--rechaza:#1baf7a}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz{color-scheme:dark;--surface:#1a1a19;
--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--total:#898781;--coord:#d95926;--apoya:#3987e5;--rechaza:#199e70}}
:root[data-theme="dark"] .viz{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--total:#898781;--coord:#d95926;--apoya:#3987e5;--rechaza:#199e70}
"""
STYLE = """
*{box-sizing:border-box}body{margin:0}
.viz{background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;padding:24px 16px 48px}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:24px;font-weight:600;margin:0}h2{font-size:18px;font-weight:600;margin:36px 0 4px}
.sub{color:var(--ink2);margin:4px 0 0}.note{color:var(--muted);font-size:13px;margin:0 0 12px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:20px}
.kpi{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:14px 16px}
.kpi .l{color:var(--ink2);font-size:13px}.kpi .v{font-size:26px;font-weight:600;margin-top:2px}
.kpi .h{color:var(--muted);font-size:12px}
.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--ink2);font-size:13px;margin:6px 0 10px}
.key{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;margin-right:6px}
.row{display:grid;grid-template-columns:minmax(220px,1.1fr) minmax(260px,1.6fr) minmax(170px,.8fr);gap:18px;
background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:14px 16px;margin-bottom:10px}
@media (max-width:820px){.row{grid-template-columns:1fr}}
.t{font-weight:600}.phr{color:var(--ink2);font-size:13px}.meta{color:var(--muted);font-size:12px;margin-top:4px}
.badge{display:inline-block;font-size:11px;font-weight:600;border:1px solid var(--axis);color:var(--ink2);
border-radius:999px;padding:0 8px;margin-left:6px;vertical-align:1px}
svg{display:block;width:100%;height:auto;overflow:visible}svg text{fill:var(--muted);font-size:11px}
.spark{position:relative}.spark svg:focus{outline:2px solid var(--apoya);outline-offset:2px;border-radius:4px}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--ring);border-radius:8px;
padding:6px 10px;font-size:12px;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;z-index:2}
.tip b{font-size:13px}.tip .d{color:var(--muted)}
.stance .lab{display:flex;justify-content:space-between;color:var(--ink2);font-size:12px;margin-top:6px}
.small{color:var(--ink2);font-size:13px;margin-top:10px}
.ops{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.op{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:14px 16px}
.op ul{margin:8px 0 0;padding-left:18px;color:var(--ink2);font-size:13px}
details{margin-top:28px}summary{cursor:pointer;color:var(--ink2)}
table{border-collapse:collapse;width:100%;margin-top:10px;font-size:13px;background:var(--surface)}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}td.n{text-align:right;font-variant-numeric:tabular-nums}
footer{color:var(--muted);font-size:12px;margin-top:32px}
"""
SCRIPT = """
document.querySelectorAll('.spark').forEach(function(box){
  var svg=box.querySelector('svg'),tip=box.querySelector('.tip'),hair=svg.querySelector('.hair');
  var data=JSON.parse(box.dataset.series),n=data.total.length,idx=n-1;
  function show(i){idx=Math.max(0,Math.min(n-1,i));var x=(idx+0.5)/n*600;
    hair.setAttribute('x1',x);hair.setAttribute('x2',x);hair.style.display='block';
    tip.replaceChildren();var b=document.createElement('b');b.textContent=data.total[idx]+' comentarios';tip.appendChild(b);
    tip.appendChild(document.createElement('br'));
    var c=document.createElement('span');c.textContent=data.coord[idx]+' de cuentas coordinadas';tip.appendChild(c);
    tip.appendChild(document.createElement('br'));
    var d=document.createElement('span');d.className='d';d.textContent=data.labels[idx];tip.appendChild(d);
    tip.style.display='block';var w=box.clientWidth,px=(idx+0.5)/n*w;
    tip.style.left=Math.min(Math.max(px-tip.offsetWidth/2,0),w-tip.offsetWidth)+'px';tip.style.top='-8px';}
  function hide(){tip.style.display='none';hair.style.display='none';}
  svg.addEventListener('pointermove',function(e){var r=svg.getBoundingClientRect();show(Math.floor((e.clientX-r.left)/r.width*n));});
  svg.addEventListener('pointerleave',hide);svg.addEventListener('focus',function(){show(idx);});svg.addEventListener('blur',hide);
  svg.addEventListener('keydown',function(e){if(e.key==='ArrowLeft'){show(idx-1);e.preventDefault();}
    if(e.key==='ArrowRight'){show(idx+1);e.preventDefault();}});
});
"""


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def area_path(values: np.ndarray, peak: float, width: float = 600, height: float = 60) -> tuple[str, str]:
    n = len(values)
    xs = (np.arange(n) + 0.5) / n * width
    ys = height - (values / peak * (height - 4) if peak > 0 else np.zeros(n))
    line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return line, f"{line} L{xs[-1]:.1f},{height} L{xs[0]:.1f},{height} Z"


def sparkline(total: np.ndarray, coord: np.ndarray, origin_x: float | None, labels: list[str], ticks: list) -> str:
    peak = max(float(total.max()), 1.0)
    t_line, t_area = area_path(total, peak)
    c_line, c_area = area_path(coord, peak)
    parts = [f'<line x1="0" y1="60" x2="600" y2="60" stroke="var(--axis)" stroke-width="1"/>']
    for x, label in ticks:
        parts.append(f'<line x1="{x:.1f}" y1="60" x2="{x:.1f}" y2="64" stroke="var(--axis)"/>'
                     f'<text x="{x:.1f}" y="76" text-anchor="middle">{esc(label)}</text>')
    parts.append(f'<path d="{t_area}" fill="var(--total)" fill-opacity=".12"/>'
                 f'<path d="{t_line}" fill="none" stroke="var(--total)" stroke-width="1.5" stroke-linejoin="round"/>')
    if coord.sum() > 0:
        parts.append(f'<path d="{c_area}" fill="var(--coord)" fill-opacity=".25"/>'
                     f'<path d="{c_line}" fill="none" stroke="var(--coord)" stroke-width="2" stroke-linejoin="round"/>')
    if origin_x is not None:
        parts.append(f'<line x1="{origin_x:.1f}" y1="0" x2="{origin_x:.1f}" y2="60" stroke="var(--ink2)" stroke-width="1"/>'
                     f'<circle cx="{origin_x:.1f}" cy="60" r="4" fill="var(--ink2)" stroke="var(--surface)" stroke-width="2"/>')
    parts.append('<line class="hair" x1="0" y1="0" x2="0" y2="60" stroke="var(--ink)" stroke-width="1" style="display:none"/>')
    parts.append(f'<text x="600" y="10" text-anchor="end">pico {fmt_int(peak)}/{BUCKET_H}h</text>')
    data = json.dumps({"total": total.astype(int).tolist(), "coord": coord.astype(int).tolist(), "labels": labels})
    return (f'<div class="spark" data-series="{esc(data)}"><div class="tip" role="status"></div>'
            f'<svg viewBox="0 0 600 80" tabindex="0" role="img" aria-label="Volumen en el tiempo">{"".join(parts)}</svg></div>')


def stance_bar(apoya: int, rechaza: int) -> str:
    total = max(apoya + rechaza, 1)
    wa = apoya / total * 200
    gap = 2 if apoya and rechaza else 0
    return (f'<div class="stance"><svg viewBox="0 0 200 12" role="img" aria-label="Postura">'
            f'<rect x="0" y="0" width="{max(wa - gap / 2, 0):.1f}" height="12" rx="4" fill="var(--apoya)"/>'
            f'<rect x="{wa + gap / 2:.1f}" y="0" width="{max(200 - wa - gap / 2, 0):.1f}" height="12" rx="4" fill="var(--rechaza)"/>'
            f'</svg><div class="lab"><span><span class="sw" style="background:var(--apoya)"></span>apoya {apoya / total:.0%}</span>'
            f'<span><span class="sw" style="background:var(--rechaza)"></span>rechaza {rechaza / total:.0%}</span></div></div>')


def write_dashboard(ctx: Context, cost: dict):
    df = ctx.comments
    offset = ctx.report.get("ingesta", {}).get("utc_offset", "-03:00")
    sign = -1 if offset.startswith("-") else 1
    tz = timezone(sign * timedelta(hours=int(offset[1:3]), minutes=int(offset[4:6])))
    to_dt = lambda e: datetime.fromtimestamp(float(e), tz=tz)  # noqa: E731

    groups = dict(ctx.sql("SELECT cuenta_id, grupo FROM grupos"))
    df = df.assign(grupo=df["cuenta_id"].map(groups).fillna(""))
    t0 = np.floor(df["t"].min() / 3600) * 3600
    span = df["t"].max() - t0
    n_buckets = int(np.ceil((span + 1) / (BUCKET_H * 3600)))
    df["b"] = ((df["t"] - t0) // (BUCKET_H * 3600)).astype(int)
    labels = [f"{to_dt(t0 + i * BUCKET_H * 3600):%d/%m %H:%M}–{to_dt(t0 + (i + 1) * BUCKET_H * 3600):%H:%M}"
              for i in range(n_buckets)]
    days = sorted({to_dt(t0 + i * BUCKET_H * 3600).date() for i in range(n_buckets)})
    step = max(1, len(days) // 7)
    ticks = []
    for d in days[::step]:
        e = datetime(d.year, d.month, d.day, tzinfo=tz).timestamp()
        if e >= t0:
            ticks.append(((e - t0) / (n_buckets * BUCKET_H * 3600) * 600, f"{d:%d/%m}"))

    narr = ctx.con.execute("SELECT narrativa, titulo, frases_clave, origen_epoch FROM narrativas").df()
    # An operation "pushes" the narrative holding at least a quarter of its on-topic comments.
    op_main = {}
    for op, g in df[(df["grupo"] != "") & (df["narrativa"] != "")].groupby("grupo"):
        counts = g["narrativa"].value_counts()
        if counts.iloc[0] / len(g) >= 0.25:
            op_main[op] = counts.index[0]
    op_size = pd.Series(groups, dtype="object").value_counts()
    first_day_end = df["t"].min() + 24 * 3600
    rows_html, table_rows = [], []
    for _, n in narr.iterrows():
        g = df[df["narrativa"] == n["narrativa"]]
        if g.empty:
            continue
        total = np.bincount(g["b"], minlength=n_buckets)[:n_buckets].astype(float)
        coord = np.bincount(g.loc[g["grupo"] != "", "b"], minlength=n_buckets)[:n_buckets].astype(float)
        apoya, rechaza = int((g["postura"] == "apoya").sum()), int((g["postura"] == "rechaza").sum())
        origin = n["origen_epoch"]
        emergent = origin is not None and origin > first_day_end
        origin_x = (origin - t0) / (n_buckets * BUCKET_H * 3600) * 600 if emergent else None
        push_txt = "; ".join(f"{esc(op)}: {op_size[op]} cuentas, {(g['grupo'] == op).mean():.1%} del volumen"
                             for op, m in op_main.items() if m == n["narrativa"]) or "ninguna operación detectada"
        typical = g.sort_values("tipicidad", ascending=False)
        ex = []
        for side in ("apoya", "rechaza"):
            s = typical[(typical["postura"] == side) & (typical["nw"] > 5)]
            if len(s):
                ex.append(f'<div class="small"><span class="sw" style="background:var(--{side})"></span>“{esc(s.iloc[0]["texto"][:140])}”</div>')
        badge = '<span class="badge">emergente</span>' if emergent else ""
        rows_html.append(
            f'<div class="row"><div><div class="t">{esc(n["titulo"])}{badge}</div>'
            f'<div class="phr">{esc(n["frases_clave"].replace("|", " · "))}</div>'
            f'<div class="meta">{esc(n["narrativa"])} · {fmt_int(len(g))} comentarios · nace {to_dt(origin):%d/%m %H:%M}</div>'
            f'{"".join(ex)}</div>'
            f'<div>{sparkline(total, coord, origin_x, labels, ticks)}</div>'
            f'<div>{stance_bar(apoya, rechaza)}<div class="small">Empujada por: {push_txt}</div></div></div>')
        table_rows.append(f"<tr><td>{esc(n['narrativa'])}</td><td>{esc(n['titulo'])}</td><td class='n'>{fmt_int(len(g))}</td>"
                          f"<td class='n'>{apoya / max(len(g), 1):.0%}</td><td class='n'>{rechaza / max(len(g), 1):.0%}</td>"
                          f"<td>{to_dt(origin):%d/%m/%Y %H:%M}</td><td class='n'>{coord.sum() / max(len(g), 1):.1%}</td></tr>")

    acc = ctx.con.execute("SELECT cuenta_id, fecha_creacion FROM cuentas").df()
    start = to_dt(df["t"].min()).date()
    acc["edad"] = (pd.Timestamp(start) - pd.to_datetime(acc["fecha_creacion"])).dt.days
    age = acc.set_index("cuenta_id")["edad"]
    titles = dict(zip(narr["narrativa"], narr["titulo"]))
    ops_html = []
    for op, members in pd.Series(groups).groupby(pd.Series(groups)).groups.items():
        g = df[df["grupo"] == op]
        hours = np.bincount([to_dt(t).hour for t in g["t"]], minlength=24)
        peak = max(hours.max(), 1)
        bars = "".join(f'<rect x="{h * 14 + 2}" y="{44 - c / peak * 40:.1f}" width="10" height="{max(c / peak * 40, 0.5):.1f}" '
                       f'rx="2" fill="var(--coord)"><title>{h:02d} h: {c} comentarios</title></rect>' for h, c in enumerate(hours))
        hour_axis = "".join(f'<text x="{h * 14 + 7}" y="58" text-anchor="middle">{h:02d}</text>' for h in (0, 6, 12, 18, 23))
        top = g.loc[g["narrativa"] != "", "narrativa"].value_counts()
        main = top.index[0] if len(top) else ""
        on_main = g[g["narrativa"] == main]
        stance_main = on_main["postura"].value_counts(normalize=True)
        tags = pd.Series([t for txt in g["texto"] for t in txt.lower().split() if t.startswith("#")]).value_counts().head(3)
        samples = on_main[on_main["nw"] > 5].drop_duplicates("texto").head(3)["texto"]
        ops_html.append(
            f'<div class="op"><div class="t">{esc(op)} · {len(members)} cuentas · {fmt_int(len(g))} comentarios</div>'
            f'<div class="small">Empuja <b>{esc(titles.get(main, main))}</b> ({esc(main)}), '
            f'{stance_main.get("apoya", 0):.0%} a favor · {len(on_main) / max(len(g), 1):.0%} de su actividad</div>'
            f'<div class="small">Antigüedad mediana de las cuentas: {age.reindex(members).median():.0f} días'
            f'{" · hashtags: " + esc(", ".join(tags.index)) if len(tags) else ""}</div>'
            f'<div class="small">Horas de actividad</div><svg viewBox="0 0 336 62" role="img" aria-label="Horas de actividad">'
            f'<line x1="0" y1="44" x2="336" y2="44" stroke="var(--axis)"/>{bars}{hour_axis}</svg>'
            f'<ul>{"".join(f"<li>“{esc(s[:120])}”</li>" for s in samples)}</ul></div>')

    noise = (df["narrativa"] == "").mean()
    period = f"{to_dt(df['t'].min()):%d/%m/%Y} al {to_dt(df['t'].max()):%d/%m/%Y}"
    kpis = [("Comentarios", fmt_int(len(df)), period), ("Narrativas", str(len(narr)), f"{noise:.0%} es ruido"),
            ("Operaciones coordinadas", str(len(set(groups.values()))), f"{fmt_int(len(groups))} cuentas"),
            ("Tiempo de proceso", f"{cost['segundos']:.0f} s", cost.get("hardware", "").split(",")[0].split(" w/")[0]),
            ("Costo de IA", f"US$ {cost['usd_ia']:.3f}".replace(".", ","),
             f"toda la corrida · US$ {cost['usd_por_mil_comentarios']:.5f}".replace(".", ",") + " cada 1.000")]
    kpi_html = "".join(f'<div class="kpi"><div class="l">{esc(l)}</div><div class="v">{esc(v)}</div><div class="h">{esc(h)}</div></div>'
                       for l, v, h in kpis)
    page = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Radar de narrativas</title><style>{COLORS}{STYLE}</style></head><body class="viz"><div class="wrap">
<h1>Radar de narrativas</h1><p class="sub">Qué circula, desde cuándo y quién lo empuja · {esc(ctx.datos.name)} · {period}</p>
<div class="kpis">{kpi_html}</div>
<h2>Qué circula y desde cuándo</h2>
<p class="note">Una fila por narrativa. El punto marca cuándo nació; las emergentes aparecieron a mitad del período.</p>
<div class="legend"><span><span class="key" style="background:var(--total)"></span>Comentarios cada {BUCKET_H} h</span>
<span><span class="key" style="background:var(--coord)"></span>De cuentas coordinadas</span>
<span><span class="sw" style="background:var(--apoya)"></span>Apoya</span><span><span class="sw" style="background:var(--rechaza)"></span>Rechaza</span></div>
{"".join(rows_html)}
<h2>Quién lo empuja</h2><p class="note">Grupos de cuentas que actúan juntas una y otra vez, más allá del azar.</p>
<div class="ops">{"".join(ops_html) or '<p class="note">No se detectaron operaciones coordinadas.</p>'}</div>
<details><summary>Ver tabla</summary><table><thead><tr><th>Id</th><th>Narrativa</th><th>Comentarios</th><th>Apoya</th>
<th>Rechaza</th><th>Origen</th><th>Coordinado</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table></details>
<footer>Generado por radar · {esc(cost.get('almacenamiento', ''))} · embeddings {esc(cost.get('modelo_embeddings', ''))}
· IA {esc(cost.get('modelo_ia') or 'no usada')}</footer></div><script>{SCRIPT}</script></body></html>"""
    (ctx.entrega / "tablero.html").write_text(page, encoding="utf-8")

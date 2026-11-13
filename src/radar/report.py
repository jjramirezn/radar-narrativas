"""Self-contained HTML report (entrega/tablero.html): what circulates, since when, and who pushes it.

Laid out like an intelligence briefing rather than a SaaS dashboard: a headline and lead paragraph
written from the data, a timeline with one lane per narrative, one file per coordinated operation
with the evidence behind the flag, and the narratives one by one. No external resources (inline
SVG, system fonts), so it opens offline. Light and dark themes; every number in a chart is also in
the table view.
"""
import html
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .context import Context

BUCKET_H = 12
WORDS = {1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete", 8: "ocho", 9: "nueve",
         10: "diez"}

CSS = """
.viz{color-scheme:light;--paper:#f6f4ef;--card:#fbfaf6;--ink:#151513;--ink2:#46453f;--muted:#85827a;
--rule:#d6d2c7;--rule2:#e9e6dd;--coord:#eb6834;--apoya:#2a78d6;--rechaza:#1baf7a}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz{color-scheme:dark;--paper:#121210;
--card:#1a1a17;--ink:#f1efe8;--ink2:#c4c1b7;--muted:#8b887f;--rule:#38372f;--rule2:#25241f;--coord:#d95926;
--apoya:#3987e5;--rechaza:#199e70}}
:root[data-theme="dark"] .viz{color-scheme:dark;--paper:#121210;--card:#1a1a17;--ink:#f1efe8;--ink2:#c4c1b7;
--muted:#8b887f;--rule:#38372f;--rule2:#25241f;--coord:#d95926;--apoya:#3987e5;--rechaza:#199e70}
*{box-sizing:border-box}body{margin:0}
.viz{background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
padding:40px 20px 64px}
.wrap{max-width:1080px;margin:0 auto}
.serif{font-family:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
.kicker{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
.masthead{display:flex;justify-content:space-between;align-items:baseline;border-bottom:2px solid var(--ink);
padding-bottom:8px;margin-bottom:28px;gap:12px;flex-wrap:wrap}
.masthead .brand{font-weight:600;letter-spacing:.02em}.masthead .meta{color:var(--muted);font-size:13px}
h1{font-size:36px;line-height:1.15;font-weight:600;margin:0 0 16px;max-width:820px;letter-spacing:-.01em}
.stats{display:flex;flex-wrap:wrap;gap:6px 22px;color:var(--ink2);font-size:13px;margin:0 0 6px}
.stats b{color:var(--ink);font-weight:600;font-size:15px;margin-right:4px}
section{border-top:1px solid var(--rule);margin-top:40px;padding-top:18px}
h2{font-size:24px;font-weight:600;margin:0 0 6px;letter-spacing:-.005em}
.note{color:var(--muted);font-size:13px;margin:0 0 14px;max-width:760px}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;color:var(--ink2);font-size:12px;margin:0 0 8px}
.sw{display:inline-block;width:10px;height:10px;vertical-align:-1px;margin-right:6px}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{fill:var(--ink2);font-size:11px;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
svg .lane{fill:var(--ink);font-size:13.5px;font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif}
svg .lid{fill:var(--muted);font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px}
svg .num{fill:var(--ink);font-variant-numeric:tabular-nums;font-size:12px}
svg .opt{fill:var(--coord);font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;font-weight:600}
svg rect.cell:hover{stroke:var(--ink);stroke-width:1}
.ops{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));border-top:1px solid var(--rule)}
.op{padding:18px 20px 20px 0;border-bottom:1px solid var(--rule)}
.op+.op{padding-left:20px;border-left:1px solid var(--rule)}
@media (max-width:700px){.op+.op{padding-left:0;border-left:0}}
.opid{font-size:12px;color:var(--coord);font-weight:600}
.big{font-size:44px;line-height:1;font-weight:600;margin:6px 0 2px}.big span{font-size:15px;font-weight:400;color:var(--ink2)}
.opclaim{font-size:16px;line-height:1.45;margin:10px 0 14px}
.ind{display:grid;grid-template-columns:max-content 1fr max-content;gap:6px 10px;align-items:center;font-size:12.5px;
color:var(--ink2);margin:0 0 14px}
.track{height:6px;background:var(--rule2);position:relative}.track i{position:absolute;inset:0 auto 0 0;background:var(--coord)}
.pctv{font-variant-numeric:tabular-nums;color:var(--ink)}
blockquote{margin:8px 0 0;padding-left:12px;border-left:2px solid var(--rule);color:var(--ink2);font-size:14px;line-height:1.45}
blockquote.a{border-left-color:var(--apoya)}blockquote.r{border-left-color:var(--rechaza)}
.nrow{display:grid;grid-template-columns:54px minmax(0,1fr) 230px;gap:16px;padding:16px 0;border-bottom:1px solid var(--rule)}
@media (max-width:760px){.nrow{grid-template-columns:44px minmax(0,1fr)}.nnum{grid-column:2}}
.nid{color:var(--muted);font-size:12px;padding-top:4px}
.nrow h3{font-size:19px;line-height:1.3;font-weight:600;margin:0 0 4px}
.phr{color:var(--muted);font-size:13px;margin:0}
.badge{display:inline-block;font:600 10.5px/1.6 system-ui,sans-serif;letter-spacing:.08em;text-transform:uppercase;
color:var(--ink);border:1px solid var(--ink);padding:0 6px;margin-left:8px;vertical-align:3px}
.nnum{font-size:13px;color:var(--ink2);line-height:1.6}.nnum b{color:var(--ink);font-size:16px;font-variant-numeric:tabular-nums}
.sbar{display:flex;height:6px;gap:2px;margin:6px 0 2px}.sbar i{display:block}
details summary{cursor:pointer;color:var(--ink2);font-size:13px;margin-top:8px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:10px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--rule2);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:11px;letter-spacing:.08em;text-transform:uppercase}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.foot{border-top:1px solid var(--rule);margin-top:40px;padding-top:12px;color:var(--muted);font-size:12px;line-height:1.7}
.foot a{color:var(--ink2)}
.ophead{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}
.ophead .big{display:flex;align-items:center;gap:10px}
.ophead .big span{white-space:nowrap;line-height:1.3}
.web{width:104px;height:104px;flex:none}
.tag{display:inline-block;font-size:11.5px;color:var(--coord);border:1px solid var(--coord);padding:0 6px;margin:0 6px 10px 0}
.nrow blockquote{max-width:680px}
"""


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def num(n) -> str:
    return f"{int(round(n)):,}".replace(",", ".")


def pct(x: float, decimals: int = 0) -> str:
    return f"{x * 100:.{decimals}f}%".replace(".", ",")


def words(n: int) -> str:
    return WORDS.get(n, str(n))


def short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip(" ,.;:") + "…"


def bare(claim: str) -> str:
    return str(claim).strip().rstrip(".")


def web_svg(web) -> str:
    """Small circular network: each dot an account, each line a pair that acted together."""
    if not web:
        return ""
    n, edges, weights = web
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    xs, ys = 59 + 52 * np.cos(angles), 59 + 52 * np.sin(angles)
    top = max(weights) if weights else 1
    lines = "".join(f'<line x1="{xs[a]:.1f}" y1="{ys[a]:.1f}" x2="{xs[b]:.1f}" y2="{ys[b]:.1f}" stroke="var(--coord)" '
                    f'stroke-width=".6" stroke-opacity="{0.15 + 0.45 * w / top:.2f}"/>' for (a, b), w in zip(edges, weights))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="var(--coord)"/>' for x, y in zip(xs, ys))
    return (f'<svg class="web" viewBox="0 0 118 118" role="img" aria-label="Red de {n} cuentas y {len(edges)} vínculos">'
            f'<title>{n} cuentas, {len(edges)} pares que actuaron juntos</title>{lines}{dots}</svg>')


def headline(n_narr: int, emergent: list, ops: dict, pushed: set) -> str:
    if emergent:
        first = ("Una narrativa nació" if len(emergent) == 1
                 else f"{words(len(emergent)).capitalize()} narrativas nacieron") + " a mitad del período"
    else:
        first = f"{words(n_narr).capitalize()} narrativas circulan desde el primer día"
    if not ops:
        return first + " y no encontramos operaciones coordinadas."
    k = len(ops)
    verb = "empuja" if k == 1 else "empujan"
    who = "una operación coordinada" if k == 1 else f"{words(k)} operaciones coordinadas"
    target = "una" if len(pushed) == 1 else words(len(pushed))
    return f"{first}; {who} {verb} {target} de las {n_narr}."


def write_dashboard(ctx: Context, cost: dict):
    df = ctx.comments
    offset = ctx.report.get("ingesta", {}).get("utc_offset", "-03:00")
    sign = -1 if offset.startswith("-") else 1
    tz = timezone(sign * timedelta(hours=int(offset[1:3]), minutes=int(offset[4:6])))
    to_dt = lambda e: datetime.fromtimestamp(float(e), tz=tz)  # noqa: E731

    groups = dict(ctx.sql("SELECT cuenta_id, grupo FROM grupos"))
    df = df.assign(grupo=df["cuenta_id"].map(groups).fillna(""))
    t_min, t_max = float(df["t"].min()), float(df["t"].max())
    t0 = np.floor(t_min / 3600) * 3600
    span = BUCKET_H * 3600
    n_b = int(np.ceil((t_max - t0 + 1) / span))
    df["b"] = ((df["t"] - t0) // span).astype(int)
    days = (to_dt(t_max).date() - to_dt(t_min).date()).days + 1

    narr = ctx.con.execute("SELECT narrativa, titulo, frases_clave, origen_epoch FROM narrativas").df()
    by_n = {n: g for n, g in df[df["narrativa"] != ""].groupby("narrativa")}
    narr = narr[narr["narrativa"].isin(by_n)].copy()
    narr["size"] = narr["narrativa"].map(lambda n: len(by_n[n]))
    narr["emergente"] = narr["origen_epoch"] > t_min + 24 * 3600
    titles = dict(zip(narr["narrativa"], narr["titulo"]))

    # Operations: what each pushes, and the evidence behind flagging it.
    acc = ctx.con.execute("SELECT cuenta_id, fecha_creacion FROM cuentas").df()
    acc["edad"] = (pd.Timestamp(to_dt(t_min).date()) - pd.to_datetime(acc["fecha_creacion"])).dt.days
    age = acc.set_index("cuenta_id")["edad"]
    op_info = {}
    for op, g in df[df["grupo"] != ""].groupby("grupo"):
        members = [a for a, o in groups.items() if o == op]
        on_topic = g[g["narrativa"] != ""]
        main, main_stance, share_main, unique = "", "", 0.0, 0.0
        if len(on_topic):
            counts = on_topic.groupby(["narrativa", "postura"]).size()
            (main, main_stance), top = counts.idxmax(), counts.max()
            unique = top / len(on_topic)
            share_main = (on_topic["narrativa"] == main).mean()
        hours = np.bincount([to_dt(t).hour for t in g["t"]], minlength=24)
        window = max(range(24), key=lambda h: sum(hours[(h + i) % 24] for i in range(4)))
        conc = sum(hours[(window + i) % 24] for i in range(4)) / max(hours.sum(), 1)
        ages = age.reindex(members).dropna()
        op_info[op] = {
            "members": len(members), "comments": len(g), "main": main, "stance": main_stance,
            "unique": unique, "share_main": share_main, "hours": hours, "window": window, "conc": conc,
            "new": float((ages <= 30).mean()) if len(ages) else 0.0, "age": float(ages.median()) if len(ages) else None,
            "density": ctx.report.get("operaciones", {}).get(op, {}).get("densidad", 0.0),
            "tags": pd.Series([w for txt in g["texto"] for w in txt.lower().split() if w.startswith("#")]).value_counts().head(3),
            "samples": ((g[(g["narrativa"] == main) & (g["postura"] == main_stance) & (g["nw"] > 5)] if main else g)
                        ["texto"].value_counts().head(2).index.tolist()),
        }
    pushed_by = {}
    for op, info in op_info.items():
        if info["main"] and info["share_main"] >= 0.25:
            pushed_by.setdefault(info["main"], []).append(op)

    # Headline, written from the numbers.
    emergent = narr[narr["emergente"]].sort_values("origen_epoch")
    title = headline(len(narr), list(emergent["narrativa"]), op_info, set(pushed_by))

    # Timeline: one lane per narrative, ordered by birth.
    lanes = narr.sort_values(["origen_epoch", "size"], ascending=[False, False])
    W, label_w, sx0, sx1, rx = 1080, 320, 335, 815, 830
    max_size = max(narr["size"].max(), 1)
    lane_h, top = 34, 44
    cw = (sx1 - sx0) / n_b
    coord_peak = max([np.bincount(g.loc[g["grupo"].isin(pushed_by.get(n, [])), "b"], minlength=n_b).max()
                      for n, g in by_n.items()] + [1])
    parts = []
    day0 = to_dt(t_min).date()
    for d in range(days + 1):
        e = datetime(day0.year, day0.month, day0.day, tzinfo=tz).timestamp() + d * 86400
        if t0 <= e <= t0 + n_b * span:
            x = sx0 + (e - t0) / (n_b * span) * (sx1 - sx0)
            parts.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{top + len(lanes) * lane_h}" '
                         f'stroke="var(--rule2)" stroke-width="1"/>')
            if d % max(1, days // 7) == 0 and x < sx1 - 36:
                parts.append(f'<text x="{x + 3:.1f}" y="{top - 14}">{(day0 + timedelta(days=d)):%d/%m}</text>')
    parts.append(f'<text x="{rx}" y="{top - 14}">comentarios · postura</text>')
    for i, (_, n) in enumerate(lanes.iterrows()):
        g = by_n[n["narrativa"]]
        y = top + i * lane_h
        sy = y + 8
        total = np.bincount(g["b"], minlength=n_b)[:n_b]
        pushers = g["grupo"].isin(pushed_by.get(n["narrativa"], []))
        coord = np.bincount(g.loc[pushers, "b"], minlength=n_b)[:n_b]
        peak = max(total.max(), 1)
        parts.append(f'<text class="lid" x="0" y="{sy + 7}" dominant-baseline="central">{esc(n["narrativa"])}</text>'
                     f'<text class="lane" x="36" y="{sy + 7}" dominant-baseline="central">{esc(short(n["titulo"], 42))}'
                     f'<title>{esc(n["titulo"])}</title></text>')
        parts.append(f'<rect x="{sx0}" y="{sy}" width="{sx1 - sx0}" height="14" fill="var(--rule2)"/>')
        for b in np.nonzero(total)[0]:
            alpha = 0.10 + 0.90 * np.sqrt(total[b] / peak)
            when = f"{to_dt(t0 + b * span):%d/%m %H:%M}–{to_dt(t0 + (b + 1) * span):%H:%M}"
            extra = f", {num(coord[b])} de cuentas coordinadas" if coord[b] else ""
            parts.append(f'<rect class="cell" x="{sx0 + b * cw:.1f}" y="{sy}" width="{max(cw - 0.8, 0.5):.1f}" height="14" '
                         f'fill="var(--ink)" fill-opacity="{alpha:.2f}"><title>{when}: {num(total[b])} comentarios{extra}'
                         f'</title></rect>')
        for b in np.nonzero(coord)[0]:
            parts.append(f'<rect x="{sx0 + b * cw:.1f}" y="{sy + 17}" width="{max(cw - 0.8, 0.5):.1f}" height="4" '
                         f'fill="var(--coord)" fill-opacity="{0.35 + 0.65 * coord[b] / coord_peak:.2f}"/>')
        if n["emergente"]:
            bx = sx0 + (n["origen_epoch"] - t0) / (n_b * span) * (sx1 - sx0)
            parts.append(f'<path d="M{bx:.1f},{sy - 1} l-4,-6 h8 z" fill="var(--ink)"/>')
            label = f"nace {to_dt(n['origen_epoch']):%d/%m %H:%M}"
            if bx - sx0 > 104:
                parts.append(f'<text x="{bx - 7:.1f}" y="{sy + 7}" text-anchor="end" dominant-baseline="central" '
                             f'style="fill:var(--ink);font-weight:600">{label}</text>')
            else:
                parts.append(f'<text x="{bx + 7:.1f}" y="{sy - 4}" style="fill:var(--ink);font-weight:600">{label}</text>')
        a, r = int((g["postura"] == "apoya").sum()), int((g["postura"] == "rechaza").sum())
        wa = 60 * a / max(a + r, 1)
        parts.append(f'<rect x="{rx}" y="{sy + 3}" width="{max(70 * len(g) / max_size, 1):.1f}" height="8" '
                     f'fill="var(--ink)" fill-opacity=".55"><title>{num(len(g))} comentarios</title></rect>'
                     f'<text class="num" x="{rx + 120}" y="{sy + 7}" text-anchor="end" dominant-baseline="central">'
                     f'{num(len(g))}</text>'
                     f'<rect x="{rx + 128}" y="{sy + 4}" width="{max(wa - 1, 0):.1f}" height="6" fill="var(--apoya)"/>'
                     f'<rect x="{rx + 128 + wa + 1:.1f}" y="{sy + 4}" width="{max(60 - wa - 1, 0):.1f}" height="6" '
                     f'fill="var(--rechaza)"><title>apoya {pct(a / max(a + r, 1))} · rechaza {pct(r / max(a + r, 1))}</title></rect>')
        if n["narrativa"] in pushed_by:
            parts.append(f'<text class="opt" x="{rx + 196}" y="{sy + 7}" dominant-baseline="central">'
                         f'{esc(" ".join(pushed_by[n["narrativa"]]))}</text>')
    height = top + len(lanes) * lane_h + 8
    timeline = (f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="Línea de tiempo de las narrativas">'
                f'{"".join(parts)}</svg>')

    # Operation files.
    ops_html = []
    for op, info in sorted(op_info.items(), key=lambda kv: -kv[1]["members"]):
        hours, peak_h = info["hours"], max(info["hours"].max(), 1)
        bars = "".join(f'<rect x="{h * 12 + 1}" y="{30 - c / peak_h * 28:.1f}" width="10" height="{max(c / peak_h * 28, 0.6):.1f}" '
                       f'fill="var(--coord)"><title>{h:02d} h: {c} comentarios</title></rect>' for h, c in enumerate(hours))
        axis = "".join(f'<text x="{h * 12 + 6}" y="44" text-anchor="middle">{h:02d}</text>' for h in (0, 6, 12, 18, 23))
        w0, w1 = info["window"], (info["window"] + 4) % 24
        verb = "Empuja" if info["stance"] == "apoya" else "Ataca"
        claim = (f'{verb} «{esc(bare(titles.get(info["main"], info["main"])))}»' if info["main"]
                 else "Sin una narrativa dominante")
        indicators = [("Red cerrada", info["density"], "pares de cuentas del grupo que actúan juntas"),
                      ("Cuentas nuevas", info["new"], "creadas hasta 30 días antes del período"),
                      ("Horario concentrado", info["conc"], f"de su actividad entre las {w0:02d} y las {w1:02d} h"),
                      ("Mensaje único", info["unique"], "de sus mensajes repiten la misma narrativa y postura")]
        ind_html = "".join(f'<span title="{esc(desc)}">{esc(name)}</span><span class="track"><i style="width:{v * 100:.0f}%">'
                           f'</i></span><span class="pctv">{pct(v)}</span>' for name, v, desc in indicators)
        tags = "".join(f'<span class="tag mono">{esc(t)}</span>' for t in info["tags"].index)
        quotes = "".join(f"<blockquote>“{esc(short(s, 120))}”</blockquote>" for s in info["samples"][:1])
        ops_html.append(
            f'<article class="op"><div class="ophead"><div><div class="opid mono">{esc(op)}</div>'
            f'<div class="big">{num(info["members"])} <span>cuentas<br>{num(info["comments"])} comentarios</span></div></div>'
            f'{web_svg((ctx.coord_webs or {}).get(op))}</div>'
            f'<p class="opclaim serif">{claim}</p>{tags and f"<div>{tags}</div>"}<div class="ind">{ind_html}</div>'
            f'<svg viewBox="0 0 288 48" role="img" aria-label="Horas de actividad">'
            f'<line x1="0" y1="30" x2="288" y2="30" stroke="var(--rule)"/>{bars}{axis}</svg>{quotes}</article>')

    # Narratives one by one.
    rows, table = [], []
    for _, n in narr.sort_values("size", ascending=False).iterrows():
        g = by_n[n["narrativa"]]
        a, r = int((g["postura"] == "apoya").sum()), int((g["postura"] == "rechaza").sum())
        tot = max(a + r, 1)
        typical = g.sort_values("tipicidad", ascending=False)
        ex = []
        for side, cls in (("apoya", "a"), ("rechaza", "r")):
            s = typical[(typical["postura"] == side) & (typical["nw"] > 5)]
            if len(s):
                ex.append(f'<blockquote class="{cls}">“{esc(short(s.iloc[0]["texto"], 180))}”</blockquote>')
        badge = '<span class="badge">emergente</span>' if n["emergente"] else ""
        push = pushed_by.get(n["narrativa"], [])
        coord_share = (g["grupo"] != "").mean()
        push_txt = (f'<br>empujada por <span class="mono" style="color:var(--coord)">{esc(" ".join(push))}</span> '
                    f'({pct(coord_share, 1)} del volumen)') if push else ""
        rows.append(
            f'<div class="nrow"><div class="nid mono">{esc(n["narrativa"])}</div><div>'
            f'<h3 class="serif">{esc(n["titulo"])}{badge}</h3>{"".join(ex)}</div>'
            f'<div class="nnum"><b>{num(len(g))}</b> comentarios<br>nace el {to_dt(n["origen_epoch"]):%d/%m a las %H:%M}'
            f'<div class="sbar"><i style="flex:{a};background:var(--apoya)"></i><i style="flex:{r};background:var(--rechaza)">'
            f'</i></div>apoya {pct(a / tot)} · rechaza {pct(r / tot)}{push_txt}</div></div>')
        table.append(f"<tr><td class='mono'>{esc(n['narrativa'])}</td><td>{esc(n['titulo'])}</td><td class='n'>{num(len(g))}</td>"
                     f"<td class='n'>{pct(a / tot)}</td><td class='n'>{pct(r / tot)}</td>"
                     f"<td class='n'>{to_dt(n['origen_epoch']):%d/%m/%Y %H:%M}</td><td class='n'>{pct(coord_share, 1)}</td></tr>")

    stages = ", ".join(f"{k} {v:.0f} s" for k, v in cost.get("etapas_segundos", {}).items())
    usd = f"{cost.get('usd_ia', 0) + cost.get('usd_ia_original_de_reutilizadas', 0):.3f}".replace(".", ",")
    ai = (f"{cost.get('llamadas_ia', 0)} llamadas nuevas y {cost.get('llamadas_ia_reutilizadas', 0)} reutilizadas del "
          f"registro, {num(cost.get('tokens_ia', 0))} tokens, US$ {usd}")
    page = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Radar de narrativas</title><style>{CSS}</style></head>
<body class="viz"><div class="wrap">
<header class="masthead"><span class="brand">Radar de narrativas</span>
<span class="meta">Informe de situación · {esc(ctx.datos.name)} · {to_dt(t_min):%d/%m/%Y} al {to_dt(t_max):%d/%m/%Y}
· procesado el {datetime.now(tz):%d/%m/%Y a las %H:%M}</span></header>
<h1 class="serif">{esc(title)}</h1>
<div class="stats"><span><b>{num(len(df))}</b>comentarios</span><span><b>{len(narr)}</b>narrativas</span>
<span><b>{len(op_info)}</b>operaciones coordinadas</span><span><b>{days}</b>días</span>
<span><b>{cost['segundos']:.0f} s</b>de proceso</span><span><b>US$ {usd}</b>de IA</span></div>

<section><h2 class="serif">Qué circula y desde cuándo</h2>
<div class="legend"><span><span class="sw" style="background:var(--ink)"></span>volumen cada {BUCKET_H} h</span>
<span><span class="sw" style="background:var(--coord)"></span>la empuja una operación</span>
<span><span class="sw" style="background:var(--apoya)"></span>apoya</span>
<span><span class="sw" style="background:var(--rechaza)"></span>rechaza</span></div>
{timeline}</section>

<section><h2 class="serif">Quién lo empuja</h2>
<p class="note">Señales estadísticas para revisar a mano, no una acusación.</p>
<div class="ops">{"".join(ops_html) or '<p class="note">No se encontraron operaciones coordinadas.</p>'}</div></section>

<section><details><summary>Ver cada narrativa con un comentario de cada lado</summary>{"".join(rows)}</details>
<details><summary>Ver todos los números en una tabla</summary><table><thead><tr><th>Id</th><th>Narrativa</th>
<th>Comentarios</th><th>Apoya</th><th>Rechaza</th><th>Nace</th><th>Coordinado</th></tr></thead>
<tbody>{"".join(table)}</tbody></table></details></section>

<footer class="foot" title="{esc(stages)} · {esc(ai)}">Vectores locales ({esc(cost.get('modelo_embeddings', ''))}) ·
narrativas por respuestas entre comentarios · {esc(cost.get('modelo_ia') or 'sin IA')} en muestras ·
{esc(cost.get('almacenamiento', ''))} · reproducible con registro_ia/<br>Datos:
<a href="narrativas.csv">narrativas.csv</a> · <a href="comentarios.csv">comentarios.csv</a> ·
<a href="cuentas.csv">cuentas.csv</a> · <a href="costo.json">costo.json</a></footer>
</div></body></html>"""
    (ctx.entrega / "tablero.html").write_text(page, encoding="utf-8")

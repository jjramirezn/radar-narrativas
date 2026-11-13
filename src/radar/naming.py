"""Name and review narratives with one Claude call.

For every narrative Claude sees its key phrases and a dozen comments covering its range
(the comment nearest each of 12 k-means centers), and returns: the claim stated as one
simple sentence the way its supporters would say it (the stance step judges comments
against it), key phrases, whether it is a narrative at all, and whether it repeats
another one. Duplicates are merged and non-narratives become noise. Without Claude, the
keyword titles from the clustering step stay.
"""
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import narratives
from .context import Context
from .llm import LLM

EXAMPLES = 12
HEADER = re.compile(r"^Grupo (\S+) \(\d+ comentarios\)\. Palabras frecuentes: (.*)$", re.MULTILINE)
CONCESSIVE = re.compile(r",?\s+(?:aunque|pero|sin embargo|no obstante|mientras que)\s+", re.IGNORECASE)

SYSTEM = """Sos analista de conversación pública argentina en redes sociales.
Te paso grupos de comentarios que un algoritmo juntó por similitud. Cada grupo debería ser una
narrativa: una afirmación o encuadre que se repite con distintas palabras. En cada grupo hay
comentarios que la apoyan y otros que la rechazan; los dos lados pertenecen a la misma narrativa.
Para cada grupo devolvé:
- afirmacion: la afirmación central en UNA oración simple de hasta 15 palabras, tal como la
  sostiene quien la instala: el reclamo, la denuncia, el rumor o la promesa que el otro lado
  desmiente (por ejemplo "La obra del hospital está parada porque se robaron la plata"). Puede ser
  una negación si el reclamo lo es (por ejemplo "No hubo consulta previa a las comunidades"); no la
  des vuelta para escribirla en positivo. Una sola idea y un solo lado: nada de "aunque", "pero",
  "sin embargo" ni matices del lado contrario, porque después cada comentario se juzga a favor o
  en contra de esta oración.
- frases_clave: de 4 a 6 palabras o frases cortas típicas del grupo.
- es_narrativa: false solo si el grupo no expresa ninguna afirmación (preguntas sueltas,
  saludos, comentarios fuera de tema).
- igual_a: el id de otro grupo que sostiene la misma afirmación de fondo (aunque con otras
  palabras, otros detalles o en otra etapa de la conversación), o null. Un grupo chico que repite
  la idea de uno grande casi siempre es un fragmento de ese grupo: marcalo igual_a el grande.
  Dos afirmaciones distintas sobre el mismo tema NO son iguales.
Respondé solo con el JSON pedido."""

SCHEMA = {
    "type": "object",
    "properties": {"grupos": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "afirmacion": {"type": "string"},
            "frases_clave": {"type": "array", "items": {"type": "string"}},
            "es_narrativa": {"type": "boolean"},
            "igual_a": {"type": ["string", "null"]},
        },
        "required": ["id", "afirmacion", "frases_clave", "es_narrativa", "igual_a"],
        "additionalProperties": False}}},
    "required": ["grupos"], "additionalProperties": False,
}


def one_sided(claim: str) -> str:
    """Cut a claim at a concessive connector: stance needs a single side to judge against."""
    cut = CONCESSIVE.split(claim, maxsplit=1)[0].strip(" ,;:")
    return (cut if len(cut.split()) >= 4 else claim).rstrip(".") + "."


def examples(X: np.ndarray, g: pd.DataFrame, seed: int) -> list[str]:
    pool = g[g["nw"] > 5].drop_duplicates("u")
    if len(pool) <= EXAMPLES:
        return pool["texto"].tolist()
    km = KMeans(n_clusters=EXAMPLES, n_init=1, random_state=seed).fit(X[pool["u"]])
    picks = []
    for c in range(EXAMPLES):
        members = np.where(km.labels_ == c)[0]
        d = ((X[pool["u"].to_numpy()[members]] - km.cluster_centers_[c]) ** 2).sum(1)
        picks.append(pool["texto"].iat[members[np.argmin(d)]])
    return picks


def blocks_of(prompt: str) -> dict:
    """Group id -> (key phrases, examples), read back from a naming prompt."""
    heads = list(HEADER.finditer(prompt))
    out = {}
    for head, nxt in zip(heads, heads[1:] + [None]):
        body = prompt[head.end():nxt.start() if nxt else len(prompt)]
        out[head.group(1)] = (set(head.group(2).split(", ")),
                              {line[4:] for line in body.split("\n") if line.startswith("  - ")})
    return out


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def near_review(prompt: str):
    """Near replay: pair each group with a recorded group sharing most key phrases or examples, reuse its review."""
    current = blocks_of(prompt)

    def near(entries: list) -> tuple:
        best = (None, 0.0, {})
        for entry in entries:
            if "grupos" not in entry["respuesta"]:
                continue
            recorded = blocks_of(entry["pedido"])
            pairs = sorted(((max(jaccard(c[0], r[0]), jaccard(c[1], r[1])), cid, rid)
                            for cid, c in current.items() for rid, r in recorded.items()), reverse=True)
            match, used = {}, set()
            for score, cid, rid in pairs:
                if score >= 0.5 and cid not in match and rid not in used:
                    match[cid] = rid
                    used.add(rid)
            covered = len(match) / max(len(current), 1)
            if covered > best[1]:
                best = (entry, covered, match)
        entry, covered, match = best
        if entry is None:
            return None, 0.0
        back = {rid: cid for cid, rid in match.items()}
        groups = [{**g, "id": back[g["id"]], "igual_a": back.get(g.get("igual_a"))}
                  for g in entry["respuesta"]["grupos"] if g.get("id") in back]
        return {"grupos": groups}, covered
    return near


def run(ctx: Context, llm: LLM) -> dict:
    """Returns {narrative: claim}. Merges duplicates and drops non-narratives in place."""
    narr = ctx.con.execute("SELECT * FROM narrativas").df()
    claims = dict(zip(narr["narrativa"], narr["titulo"]))
    if not llm.available or narr.empty:
        return claims
    df, X, seed = ctx.comments, ctx.embeddings, ctx.settings.seed
    blocks = []
    for _, n in narr.iterrows():
        g = df[df["narrativa"] == n["narrativa"]]
        listing = "\n".join(f"  - {t[:200]}" for t in examples(X, g, seed))
        blocks.append(f"Grupo {n['narrativa']} ({len(g)} comentarios). Palabras frecuentes: "
                      f"{n['frases_clave'].replace('|', ', ')}\n{listing}")
    prompt = "\n\n".join(blocks)
    answer = llm.json(SYSTEM, prompt, SCHEMA, max_tokens=8000, near=near_review(prompt))
    if not answer:
        return claims

    known = set(narr["narrativa"])
    review = {g["id"]: g for g in answer.get("grupos", []) if g.get("id") in known}
    merges, dropped = {}, []
    for nid, g in review.items():
        target = g.get("igual_a")
        if target in known and target != nid and review.get(target, {}).get("igual_a") != nid:
            merges[nid] = target
        elif not g.get("es_narrativa", True):
            dropped.append(nid)
    # Follow merge chains to their end (a -> b -> c becomes a -> c).
    for nid in list(merges):
        seen = {nid}
        while merges[nid] in merges and merges[nid] not in seen:
            seen.add(merges[nid])
            merges[nid] = merges[merges[nid]]
    df["narrativa"] = df["narrativa"].replace(merges)
    df.loc[df["narrativa"].isin(dropped), "narrativa"] = ""

    rows = []
    for _, n in narr.iterrows():
        nid = n["narrativa"]
        if nid in merges or nid in dropped:
            continue
        g = review.get(nid, {})
        origin = n["origen_epoch"]
        if nid in merges.values():  # a merged narrative starts with the earliest of its parts
            members = df[df["narrativa"] == nid]
            core = members[members["tipicidad"] >= members["tipicidad"].median()]
            origin = narratives.origin(core["t"].to_numpy() if len(core) >= narratives.ORIGIN_BURST
                                       else members["t"].to_numpy())
        rows.append({"narrativa": nid, "titulo": one_sided(g["afirmacion"]) if g.get("afirmacion") else n["titulo"],
                     "frases_clave": "|".join(g.get("frases_clave") or []) or n["frases_clave"],
                     "origen_epoch": origin})
    ctx.put_table("narrativas", pd.DataFrame(rows))
    ctx.report["nombres"] = {"fusionadas": merges, "descartadas_como_ruido": dropped}
    return {r["narrativa"]: r["titulo"] for r in rows}

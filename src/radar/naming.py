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
    answer = llm.json(SYSTEM, "\n\n".join(blocks), SCHEMA, max_tokens=8000)
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

"""Stance: does each comment support or reject its narrative's claim?

Per narrative, a small classifier learns from ~60 labeled comments and labels the rest
(measured on practice: 0.99 macro-F1 with 60 correct labels, see eval/probe_label_selection.py):
1. Round 1: k-means splits the narrative's comments into 30 groups by meaning; the comment
   nearest each center gets labeled: supports, rejects, or off-topic.
2. Round 2: if round 1 came back lopsided (minority under 25%), 30 more where a first
   classifier is least sure, since that's where the rare side hides; otherwise 30 more
   diverse ones.
3. A logistic regression on the embeddings labels every other comment of the narrative.
4. Replies: short agreements ("Tal cual", "Exacto") take the stance of the comment they
   answer; replies the classifier is unsure about take the opposite stance, since most
   replies argue back.
Labels come from Claude. Without it, each narrative is split along the direction in which
replies differ from their parents, and the side using more rejection words is "rechaza".
"""
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from .context import Context
from .llm import LLM

ROUND = 30
LOPSIDED = 0.25
C = 10.0
SHORT_REPLY_WORDS = 5
UNSURE = 0.15  # |p - 0.5| below this counts as unsure
MIN_SUPPORT = 0.40  # under this share of "apoya", the claim was written from the minority side
OP_AGAINST = 0.80   # an operation writing this much against its own narrative means the claim is backwards
MAX_OTHER = 0.25    # relabeling a flipped narrative is kept only if under this share of labels are "otra"
# Short replies that contradict their parent ("Mentira", "Nada que ver") take the opposite stance.
SHORT_DISAGREE = (r"\b(?:mentira|falso|fake|verso|chamuyo|nada que ver|para nada|cualquiera|ni ah[ií]|"
                  r"no es as[ií]|no es cierto|es al rev[eé]s|discrepo|no coincido|no estoy de acuerdo)\b|^\s*no[.!]*\s*$")
REJECTION_MARKERS = ["mentira", "falso", "fake", "verso", "trucho", "nada que ver", "no es cierto", "desmint",
                     "sin pruebas", "inventan", "invento", "no hubo", "nadie va a", "dejen de", "no es así", "exager",
                     "operación", "opereta", "es al revés"]

SYSTEM = """Sos un anotador experto en conversación pública argentina en redes sociales.
Recibís una afirmación y una lista numerada de comentarios. Para cada comentario decidí su postura
frente a ESA afirmación:
- "apoya": la afirma, la respalda o da argumentos a favor.
- "rechaza": la niega, la refuta o da argumentos en contra, aunque use palabras parecidas
  (por ejemplo "es mentira que...") o hable con ironía.
- "otra": el comentario no trata sobre esta afirmación.
Juzgá el sentido, no las palabras sueltas:
- Si la afirmación tiene dos partes, juzgá la idea principal (la primera), no un detalle.
- Cuidado con la ironía: citar lo que dice el otro lado para burlarse ("dicen que...", "según
  ellos...") suele ir en contra de lo citado.
- Defender a quien la afirmación acusa, o dar una explicación alternativa, es rechazarla.
Respondé solo con el JSON pedido."""

SCHEMA = {
    "type": "object",
    "properties": {"etiquetas": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "postura": {"type": "string", "enum": ["apoya", "rechaza", "otra"]}},
        "required": ["id", "postura"], "additionalProperties": False}}},
    "required": ["etiquetas"], "additionalProperties": False,
}


def diverse(X: np.ndarray, pool: np.ndarray, k: int, seed: int, exclude=()) -> np.ndarray:
    """The row nearest each of k k-means centers, skipping rows already chosen.

    (Skipping near-copies of chosen rows was tried and hurt: in this embedding space genuinely
    different comments often sit above 0.95 cosine, so picks drifted to atypical comments.)
    """
    pool = np.setdiff1d(pool, np.asarray(exclude, dtype=int))
    if len(pool) <= k:
        return pool
    km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(X[pool])
    picks = []
    for c in range(k):
        members = pool[km.labels_ == c]
        if len(members):
            picks.append(members[np.argmin(((X[members] - km.cluster_centers_[c]) ** 2).sum(1))])
    return np.array(picks, dtype=int)


def numbered(body: str) -> list[str]:
    """Items of a "1. ...\\n2. ..." listing, read back from a prompt (an item may span lines)."""
    texts, rest, i = [], body.removeprefix("1. "), 1
    while (cut := rest.find(f"\n{i + 1}. ")) >= 0:
        texts.append(rest[:cut])
        rest = rest[cut + len(f"\n{i + 1}. "):]
        i += 1
    return texts + [rest]


def recorded_labels(entries: list) -> dict:
    """(claim, comment) -> label, from every recorded labeling answer."""
    out = {}
    for entry in entries:
        head, sep, body = entry["pedido"].partition("\n\nComentarios:\n")
        if not sep or "etiquetas" not in entry["respuesta"]:
            continue
        claim, texts = head.removeprefix("Afirmación: "), numbered(body)
        for item in entry["respuesta"]["etiquetas"]:
            if 1 <= item.get("id", 0) <= len(texts):
                out[(claim, texts[item["id"] - 1])] = item.get("postura")
    return out


def llm_labeler(llm: LLM):
    """Returns a function (claim, texts) -> list of 'apoya' / 'rechaza' / 'otra' / None, batched per call."""
    def near_labels(claim: str, texts: list[str]):
        def near(entries: list) -> tuple:
            known = recorded_labels(entries)
            found = [{"id": i + 1, "postura": known[(claim, t)]} for i, t in enumerate(texts) if (claim, t) in known]
            return ({"etiquetas": found} if found else None), len(found) / max(len(texts), 1)
        return near

    def label(jobs: list[tuple[str, list[str]]]) -> list[list]:
        prompts = []
        for claim, texts in jobs:
            listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
            prompts.append((SYSTEM, f"Afirmación: {claim}\n\nComentarios:\n{listing}", SCHEMA, 8000,
                            near_labels(claim, texts)))
        out = []
        for (claim, texts), answer in zip(jobs, llm.json_many(prompts)):
            labels = [None] * len(texts)
            for item in (answer or {}).get("etiquetas", []):
                if 1 <= item.get("id", 0) <= len(texts):
                    labels[item["id"] - 1] = item.get("postura")
            out.append(labels)
        return out
    return label


def reply_axis_split(X: np.ndarray, rows: np.ndarray, texts: list[str], reply_pairs: np.ndarray) -> np.ndarray:
    """No-AI fallback: True = supports. Direction where replies differ from parents, polarity by rejection words."""
    if len(reply_pairs) >= 10:
        diffs = X[reply_pairs[:, 0]] - X[reply_pairs[:, 1]]
        axis = np.linalg.svd(diffs - diffs.mean(0), full_matrices=False)[2][0]
    else:
        axis = np.linalg.svd(X[rows] - X[rows].mean(0), full_matrices=False)[2][0]
    proj = X[rows] @ axis
    side = proj > np.median(proj)
    marks = np.array([sum(m in t for m in REJECTION_MARKERS) for t in texts])
    if marks[side].mean() > marks[~side].mean():
        side = ~side
    return side


def run(ctx: Context, llm: LLM, claims: dict, labeler=None, only=None) -> pd.DataFrame:
    """Label stance for every narrative, or just those in `only` (leaving the rest untouched)."""
    df, X, seed = ctx.comments, ctx.embeddings, ctx.settings.seed
    labeler = labeler or (llm_labeler(llm) if llm.available else None)
    redo = df["narrativa"].isin(only) if only else pd.Series(True, index=df.index)
    if only is None or "postura" not in df:
        df["postura"] = ""
        df["p_apoya"] = np.nan
    else:
        df.loc[redo, "postura"] = ""
        df.loc[redo, "p_apoya"] = np.nan
    narratives = sorted(n for n in df["narrativa"].unique() if n and (only is None or n in only))
    pos = pd.Series(np.arange(len(df)), index=df["comentario_id"])
    parent_pos = df["respuesta_a"].map(pos)

    # Candidate unique texts per narrative: skip short replies (the reply rule covers them).
    cands = {}
    for n in narratives:
        g = df[(df["narrativa"] == n) & ~(df["respuesta_a"].notna() & (df["nw"] <= SHORT_REPLY_WORDS))]
        cands[n] = np.unique(g["u"].to_numpy())
    text_of_u = df.drop_duplicates("u").set_index("u")["texto"]

    stats = {"etiquetadas": 0, "otra": 0, "modo": "ia" if labeler else "sin_ia"}
    if labeler:
        labeled = {n: {} for n in narratives}
        first = {n: diverse(X, cands[n], ROUND, seed) for n in narratives}
        answers = labeler([(claims.get(n, n), [text_of_u[u] for u in first[n]]) for n in narratives])
        for n, picks, labels in zip(narratives, first.values(), answers):
            labeled[n].update({u: lab for u, lab in zip(picks, labels) if lab})
        second_jobs = []
        for n in narratives:
            got = [v for v in labeled[n].values() if v in ("apoya", "rechaza")]
            share = min(got.count("apoya"), got.count("rechaza")) / max(len(got), 1)
            done = list(labeled[n])
            if share < LOPSIDED and len(set(got)) == 2:
                us = np.array([u for u, v in labeled[n].items() if v in ("apoya", "rechaza")])
                ys = np.array([labeled[n][u] == "apoya" for u in us])
                rest = np.setdiff1d(cands[n], done)
                proba = LogisticRegression(C=C, max_iter=2000).fit(X[us], ys).predict_proba(X[rest])[:, 1]
                picks = rest[np.argsort(np.abs(proba - 0.5))[:ROUND]]
            else:
                picks = diverse(X, cands[n], 2 * ROUND, seed + 1, exclude=done)[:ROUND]
            second_jobs.append((n, picks))
        answers = labeler([(claims.get(n, n), [text_of_u[u] for u in picks]) for n, picks in second_jobs])
        for (n, picks), labels in zip(second_jobs, answers):
            labeled[n].update({u: lab for u, lab in zip(picks, labels) if lab})
        # Every AI label, kept for auditing: which comments the model saw and what it said.
        pd.DataFrame([{"narrativa": n, "afirmacion": claims.get(n, n), "texto": text_of_u[u], "etiqueta": v}
                      for n in narratives for u, v in labeled[n].items()]
                     ).to_csv(ctx.trabajo / "etiquetas_ia.csv", index=False, mode="a" if only else "w",
                              header=not only)

    for n in narratives:
        rows = df.index[df["narrativa"] == n].to_numpy()
        us = df.loc[rows, "u"].to_numpy()
        if labeler:
            lab = {u: v for u, v in labeled[n].items() if v in ("apoya", "rechaza")}
            stats["etiquetadas"] += len(labeled[n])
            stats["otra"] += sum(v == "otra" for v in labeled[n].values())
            train_u = np.array(list(lab), dtype=int)
            y = np.array([lab[u] == "apoya" for u in train_u])
            proba = None
            if len(set(y)) == 2:
                proba = LogisticRegression(C=C, max_iter=2000).fit(X[train_u], y).predict_proba(X[us])[:, 1]
            elif len(y) >= 8:  # consistently one-sided
                proba = np.full(len(rows), 1.0 if y[0] else 0.0)
        else:
            proba = None
        if proba is None:  # no AI, or it returned too few labels: split along the reply direction
            stats["sin_etiquetas"] = stats.get("sin_etiquetas", 0) + 1
            replies = rows[df.loc[rows, "respuesta_a"].notna().to_numpy()]
            par = parent_pos.loc[replies].dropna().astype(int)
            pairs = np.stack([df.loc[par.index, "u"].to_numpy(), df.loc[par.to_numpy(), "u"].to_numpy()], axis=1) \
                if len(par) else np.empty((0, 2), int)
            uniq = np.unique(us)
            side = reply_axis_split(X, uniq, [str(text_of_u[u]).lower() for u in uniq], pairs)
            proba = pd.Series(side.astype(float), index=uniq).loc[us].to_numpy()
        df.loc[rows, "p_apoya"] = proba
    df.loc[df["narrativa"] != "", "postura"] = np.where(df.loc[df["narrativa"] != "", "p_apoya"] >= 0.5,
                                                        "apoya", "rechaza")

    # Reply rules, applied down the chains: parents settle before their replies.
    is_reply = df["respuesta_a"].notna() & (df["narrativa"] != "")
    short = is_reply & (df["nw"] <= SHORT_REPLY_WORDS)
    disagree = short & df["texto"].str.lower().str.contains(SHORT_DISAGREE, regex=True)
    agree = short & ~disagree
    unsure = is_reply & ~short & ((df["p_apoya"] - 0.5).abs() < UNSURE)
    flip = {"apoya": "rechaza", "rechaza": "apoya"}
    for _ in range(20):
        parent_stance = parent_pos.map(lambda p: df["postura"].iat[int(p)] if pd.notna(p) else "")
        new = df["postura"].copy()
        new[agree] = parent_stance[agree].where(parent_stance[agree] != "", df["postura"][agree])
        new[disagree | unsure] = parent_stance[disagree | unsure].map(flip).fillna(df["postura"][disagree | unsure])
        if new.equals(df["postura"]):
            break
        df["postura"] = new
    ctx.report["postura" if only is None else "postura_reetiquetada"] = stats
    return df


RESTATE_SYSTEM = """Sos analista de conversación pública argentina en redes sociales.
Te paso afirmaciones que quedaron escritas desde el lado minoritario de su discusión, junto con comentarios
del lado mayoritario. Para cada una escribí la afirmación central del lado mayoritario, en UNA oración simple
de hasta 15 palabras, como la diría quien la sostiene. Una sola idea y un solo lado: nada de "no solo",
"aunque" ni "pero". Respondé solo con el JSON pedido."""

TITLED = re.compile(r"^(\S+): «(.*)»$", re.MULTILINE)
RESTATE_SCHEMA = {
    "type": "object",
    "properties": {"afirmaciones": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "string"}, "afirmacion": {"type": "string"}},
        "required": ["id", "afirmacion"], "additionalProperties": False}}},
    "required": ["afirmaciones"], "additionalProperties": False,
}


def fix_polarity(ctx: Context, llm: LLM) -> list:
    """Flip narratives whose claim ended up on the minority side of their own conversation.

    The challenge's convention, true for all 13 practice narratives: a narrative is the claim at least half
    of its comments support, and coordinated operations push the narrative, not its rebuttal. If a narrative
    ends under 40% support, or an operation writes 80% or more against its own narrative, the claim was
    written from the other side: flip the labels and restate the claim. It never triggers on practice.
    """
    df = ctx.comments
    groups = dict(ctx.sql("SELECT cuenta_id, grupo FROM grupos"))
    on_topic = df[df["narrativa"] != ""]
    support = on_topic.groupby("narrativa")["postura"].apply(lambda s: (s == "apoya").mean())
    flip = set(support[support < MIN_SUPPORT].index)
    by_op = on_topic.assign(grupo=on_topic["cuenta_id"].map(groups)).dropna(subset=["grupo"])
    pushed = {}  # narrative -> the operation's own messages, which state the side it pushes
    for _, g in by_op.groupby("grupo"):
        main = g["narrativa"].value_counts().idxmax()
        on_main = g[g["narrativa"] == main]
        if (on_main["postura"] == "rechaza").mean() >= OP_AGAINST:
            flip.add(main)
            pushed[main] = on_main.loc[on_main["nw"] > 5, "texto"].value_counts().head(8).index.tolist()
    if not flip:
        return []
    narr = ctx.con.execute("SELECT * FROM narrativas").df()
    titles = dict(zip(narr["narrativa"], narr["titulo"]))
    flipped = sorted(flip)
    # Invert first: the cheap fix, kept unless relabeling against a restated claim clearly does better.
    mask = df["narrativa"].isin(flip)
    df.loc[mask, "postura"] = df.loc[mask, "postura"].map({"apoya": "rechaza", "rechaza": "apoya"})
    df.loc[mask, "p_apoya"] = 1 - df.loc[mask, "p_apoya"]
    inverted = df.loc[mask, ["postura", "p_apoya"]].copy()
    blocks = []
    for n in flipped:
        side = df[(df["narrativa"] == n) & (df["postura"] == "apoya") & (df["nw"] > 5)]
        examples = pushed.get(n) or side.sort_values("tipicidad", ascending=False).drop_duplicates("u").head(8)["texto"]
        blocks.append(f"{n}: «{titles[n]}»\nComentarios del lado mayoritario:\n"
                      + "\n".join(f"  - {t[:200]}" for t in examples))
    prompt = "\n\n".join(blocks)

    def near(entries: list) -> tuple:  # same claim restated before, with other examples
        want = dict(TITLED.findall(prompt))
        for entry in entries:
            title_of = dict(TITLED.findall(entry["pedido"])) if "afirmaciones" in entry["respuesta"] else {}
            by_title = {title_of[a["id"]]: a["afirmacion"] for a in entry["respuesta"].get("afirmaciones", [])
                        if a.get("id") in title_of}
            found = [{"id": n, "afirmacion": by_title[t]} for n, t in want.items() if t in by_title]
            if found:
                return {"afirmaciones": found}, len(found) / len(want)
        return None, 0.0
    answer = llm.json(RESTATE_SYSTEM, prompt, RESTATE_SCHEMA, max_tokens=2000, near=near) if llm.available else None
    restated = {a["id"]: a["afirmacion"] for a in (answer or {}).get("afirmaciones", []) if a.get("id") in flip}
    relabeled = []
    for n, claim in restated.items():
        run(ctx, llm, {**titles, n: claim}, only={n})
        g = df[df["narrativa"] == n]
        stats = ctx.report.get("postura_reetiquetada", {})
        other = stats.get("otra", 0) / max(stats.get("etiquetadas", 0), 1)
        if (g["postura"] == "apoya").mean() >= MIN_SUPPORT and other <= MAX_OTHER:
            relabeled.append(n)
        else:  # relabeling did worse (ambiguous claim): keep the inverted labels
            df.loc[g.index, ["postura", "p_apoya"]] = inverted.loc[g.index]
    def negated(claim: str) -> str:
        return f"No es cierto que {claim[:1].lower()}{claim[1:]}"

    narr["titulo"] = [restated[n] if n in relabeled else (negated(t) if n in flip else t)
                      for n, t in zip(narr["narrativa"], narr["titulo"])]
    ctx.put_table("narrativas", narr)
    ctx.report["polaridad"] = {"invertidas": flipped, "reetiquetadas": relabeled,
                               "apoyo_antes": {n: round(float(support.get(n, 0)), 3) for n in flipped}}
    return flipped

"""Sentence embeddings for unique texts, cached on disk per model.

Only texts missing from the cache get embedded, so reruns and repeated phrases are
free. The model loads from the local Hugging Face cache when it's there, so a run on
the final set never depends on the network.
"""
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# e5 models expect this prefix for symmetric tasks like clustering and classification.
PREFIX = {"intfloat/": "query: "}


def _cache_file(cache_dir: Path, model_name: str) -> Path:
    return cache_dir / f"emb_{re.sub(r'[^a-zA-Z0-9]+', '_', model_name)}.parquet"


def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, device="cpu", local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached yet: download it
        return SentenceTransformer(model_name, device="cpu")


def embed_texts(texts: list[str], model_name: str, cache_dir: Path, batch_size: int = 128) -> np.ndarray:
    """Return one L2-normalized float32 vector per text, in order."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_file(cache_dir, model_name)
    known_texts, known_vecs = [], None
    if path.exists():
        table = pq.read_table(path)
        known_texts = table.column("texto").to_pylist()
        flat = table.column("vec").combine_chunks().flatten().to_numpy()
        known_vecs = flat.reshape(len(known_texts), -1).astype(np.float32)
    index = {t: i for i, t in enumerate(known_texts)}

    missing = sorted({t for t in texts if t not in index})
    if missing:
        print(f"[radar]   embebiendo {len(missing)} textos nuevos con {model_name}", flush=True)
        prefix = next((p for k, p in PREFIX.items() if model_name.startswith(k)), "")
        model = _load_model(model_name)
        new = model.encode([prefix + t for t in missing], batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        known_vecs = new if known_vecs is None else np.vstack([known_vecs, new])
        for t in missing:
            index[t] = len(known_texts)
            known_texts.append(t)
        dim = known_vecs.shape[1]
        vec_col = pa.FixedSizeListArray.from_arrays(pa.array(known_vecs.ravel(), pa.float32()), dim)
        pq.write_table(pa.table({"texto": known_texts, "vec": vec_col}), path)
    return known_vecs[[index[t] for t in texts]]

#!/usr/bin/env python3
# Manthan/scripts/context_encoder.py
from __future__ import annotations
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text as sa_text, inspect as sa_inspect

# Optional, for nicer runtime UX
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_DB_URL = os.getenv("MANTHAN_DB_URL", "sqlite:///data/manthan.db")

# -----------------------------
# Embedding utilities
# -----------------------------
def _load_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Try sentence-transformers, else fall back to TF-IDF.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        logging.info(f"Loaded encoder: {model_name}")
        return ("sbert", model)
    except Exception as e:
        logging.warning(f"SentenceTransformer unavailable ({e}). Falling back to TF-IDF.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        return ("tfidf", TfidfVectorizer(max_features=4096))

# In scripts/context_encoder.py

def _fit_transform(encoder_kind: str, encoder, texts: List[str]) -> np.ndarray:
    if encoder_kind == "sbert":
        logging.info("Encoding with SentenceTransformer (single-process mode)...")
        # --- UPDATED: Added pool=None to disable multiprocessing and prevent deadlocks ---
        return np.asarray(
            encoder.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=128, # Process in batches for better memory management
                pool=None # This is the critical fix
            )
        )
    else:
        # ... (the TF-IDF part remains the same)
        logging.info("Fitting TF-IDF vectorizer on corpus...")
        arr = encoder.fit_transform(texts)
        return arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)

def _save_index(out_dir: Path, vectors: np.ndarray, meta: List[Dict[str, Any]], kind: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", vectors)
    with open(out_dir / "items.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logging.info(f"Wrote vectors → {out_dir/'embeddings.npy'} and metadata → {out_dir/'items.json'}")
    if kind == "sbert":
        try:
            import faiss  # type: ignore
            index = faiss.IndexFlatIP(vectors.shape[1])  # cosine if normalized
            index.add(vectors.astype(np.float32))
            faiss.write_index(index, str(out_dir / "faiss.index"))
            logging.info(f"FAISS index written → {out_dir/'faiss.index'}")
        except Exception as e:
            logging.info(f"FAISS not available ({e}); skipped writing faiss.index")

# -----------------------------
# Database
# -----------------------------
def _connect(db_url: str):
    return create_engine(db_url, future=True)

def _safe(v, default="NA"):
    if v is None:
        return default
    try:
        if isinstance(v, float) and np.isnan(v):
            return default
    except Exception:
        pass
    return v

def _first_present(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and pd.notna(d[k]):
            return d[k]
    return default

def build_context_strings(db_url: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    eng = _connect(db_url)
    logging.info(f"Reading from database… {db_url}")

    with eng.connect() as conn:
        insp = sa_inspect(conn)
        tables = set(insp.get_table_names())
        logging.info(f"Tables present: {sorted(tables)}")

        if "species" not in tables:
            logging.warning("No 'species' table found.")
            return [], []

        # Read species
        sp = pd.read_sql(sa_text("SELECT * FROM species"), conn)
        logging.info(f"Loaded species rows: {len(sp)}")

        if sp.empty:
            return [], []

        # Find key column (species_key preferred, fallback to id)
        sp_key_col = "species_key" if "species_key" in sp.columns else ("id" if "id" in sp.columns else None)
        if sp_key_col is None:
            logging.error("Neither 'species_key' nor 'id' present in 'species' table.")
            return [], []

        # Traits (if present)
        tr = pd.DataFrame()
        if "species_traits" in tables:
            tr = pd.read_sql(sa_text("SELECT * FROM species_traits"), conn)
            logging.info(f"Loaded trait rows: {len(tr)}")
        else:
            logging.info("No 'species_traits' table found; proceeding without traits.")

    # Normalize column names we’ll likely use
    # Your earlier schema showed: canonical_name, scientific_name, family, genus, class_name, order_name, kingdom, phylum, iucn_category
    # We’ll tolerate absent ones.
    needed_cols = [
        sp_key_col, "canonical_name", "scientific_name", "family", "genus",
        "class_name", "order_name", "kingdom", "phylum", "iucn_category"
    ]
    for c in needed_cols:
        if c not in sp.columns:
            sp[c] = pd.NA

    # Build traits wide (on species_key/id)
    traits_wide = pd.DataFrame()
    if not tr.empty:
        # Try to infer FK column name
        tr_key_col = "species_key" if "species_key" in tr.columns else ("species_id" if "species_id" in tr.columns else None)
        if tr_key_col is None:
            logging.warning("Traits table missing species FK column ('species_key' or 'species_id'). Skipping trait join.")
        elif "trait_name" not in tr.columns or "trait_value" not in tr.columns:
            logging.warning("Traits table missing 'trait_name'/'trait_value'. Skipping trait join.")
        else:
            traits_wide = (
                tr.pivot_table(
                    index=tr_key_col,
                    columns="trait_name",
                    values="trait_value",
                    aggfunc="first",
                )
                .reset_index()
                .rename(columns={tr_key_col: sp_key_col})
            )

            # Limit log noise: list a few trait columns
            sample_trait_cols = [c for c in traits_wide.columns if c != sp_key_col][:10]
            logging.info(f"Traits (wide) rows: {len(traits_wide)} | example columns: {sample_trait_cols}")

    # Merge species + traits (left)
    merged = sp.merge(traits_wide, on=sp_key_col, how="left") if not traits_wide.empty else sp.copy()
    logging.info(f"Merged rows: {len(merged)}")

    # Build texts
    items: List[Dict[str, Any]] = []
    texts: List[str] = []

    # Choose a few trait names you care about; they may or may not exist
    trait_keys = [
        "canopy_layer",
        "successional_role",
        "drought_tolerance",
        "soil_type_preference",
        "water_need",
        "root_depth",
    ]

    logging.info("Building context strings…")
    for _, r in tqdm(list(merged.iterrows()), total=len(merged), desc="Compose", unit="sp"):
        row = r.to_dict()

        species_key = int(row[sp_key_col]) if pd.notna(row[sp_key_col]) else None
        sci = _first_present(row, ["scientific_name", "canonical_name"], default=None)
        name = sci or f"sp_{species_key if species_key is not None else 'unknown'}"

        # Taxonomy line
        taxo = f"family={_safe(row.get('family'))}, genus={_safe(row.get('genus'))}, class={_safe(row.get('class_name'))}, order={_safe(row.get('order_name'))}, kingdom={_safe(row.get('kingdom'))}, phylum={_safe(row.get('phylum'))}"

        # IUCN
        iucn = _safe(row.get("iucn_category"))

        # Traits line (only those present)
        trait_bits = []
        for tk in trait_keys:
            if tk in merged.columns:
                trait_bits.append(f"{tk}={_safe(row.get(tk))}")
        traits_line = "traits: " + (", ".join(trait_bits) if trait_bits else "NA")

        parts = [
            f"name: {name}",
            f"iucn={iucn}",
            f"taxonomy: {taxo}",
            traits_line,
        ]
        ctx = " | ".join(parts)

        texts.append(ctx)
        items.append({
            "species_key": species_key,
            "name": name,
            "scientific_name": sci,
            "iucn_category": row.get("iucn_category"),
        })

    return items, texts

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build semantic embeddings for species in the knowledge base.")
    ap.add_argument("--db-url", default=DEFAULT_DB_URL, help="SQLAlchemy DB URL (e.g., sqlite:///data/manthan.db)")
    ap.add_argument("--out-dir", default="data/embeddings", help="Directory to write embeddings + items")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformers model id")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of species (debugging)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before limiting (for a random sample)")
    args = ap.parse_args()

    items, texts = build_context_strings(args.db_url)

    if not items:
        logging.warning("No items found in DB — did you run ingest first?")
        return

    if args.shuffle:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(items))
        items = [items[i] for i in idx]
        texts = [texts[i] for i in idx]

    if args.limit is not None:
        items = items[: args.limit]
        texts = texts[: args.limit]
        logging.info(f"Using a limited sample: {len(items)} items")

    kind, encoder = _load_encoder(args.model)
    logging.info(f"Encoding {len(texts)} context strings…")
    vectors = _fit_transform(kind, encoder, texts)

    _save_index(Path(args.out_dir), vectors, items, kind)
    logging.info("Context encoding complete.")

if __name__ == "__main__":
    main()

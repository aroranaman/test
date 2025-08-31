# Manthan/scripts/setup_knowledge_base.py
from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
DB_URL = os.getenv("MANTHAN_DB_URL", "sqlite:///data/manthan.db")

def _ensure_sqlite_dir(db_url: str) -> None:
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "", 1)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

def _is_postgres(db_url: str) -> bool:
    return db_url.startswith("postgresql://") or db_url.startswith("postgresql+")

def ensure_schema(engine: Engine) -> None:
    """Creates required tables if they don't exist. Safe to call repeatedly."""
    species_sql = (
        """
        CREATE TABLE IF NOT EXISTS species (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scientific_name TEXT NOT NULL UNIQUE,
            common_name_en TEXT,
            iucn_status TEXT,
            native_to_india INTEGER NOT NULL DEFAULT 0
        );
        """
        if engine.dialect.name == "sqlite"
        else """
        CREATE TABLE IF NOT EXISTS species (
            id SERIAL PRIMARY KEY,
            scientific_name TEXT NOT NULL UNIQUE,
            common_name_en TEXT,
            iucn_status TEXT,
            native_to_india BOOLEAN NOT NULL DEFAULT FALSE
        );
        """
    )
    traits_sql = (
        """
        CREATE TABLE IF NOT EXISTS species_traits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            species_id INTEGER NOT NULL REFERENCES species(id) ON DELETE CASCADE,
            trait_name TEXT NOT NULL,
            trait_value TEXT,
            unit TEXT
        );
        """
        if engine.dialect.name == "sqlite"
        else "..." # Simplified for brevity
    )
    # ... (Add similar SQL for interactions and economic tables)

    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        conn.exec_driver_sql(species_sql)
        conn.exec_driver_sql(traits_sql)
        # ...
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_species_name ON species(scientific_name)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_traits_species ON species_traits(species_id)")

def main():
    """Creates the database and schema."""
    _ensure_sqlite_dir(DB_URL)
    engine = create_engine(DB_URL, future=True)
    logging.info(f"Using database: {engine.url}")
    ensure_schema(engine)
    logging.info("✅ Knowledge base schema setup complete.")

if __name__ == "__main__":
    main()
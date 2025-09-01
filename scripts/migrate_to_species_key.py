#!/usr/bin/env python3
# scripts/migrate_to_species_key.py
from __future__ import annotations
import sqlite3
from pathlib import Path

DB_PATH = Path("data/manthan.db")

def colnames(cur, table):
    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]

def has_table(cur, name):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

def main():
    if not DB_PATH.exists():
        print(f"❌ DB not found: {DB_PATH}")
        return
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # 1) species: ensure PK is species_key (and drop legacy id if present)
    if has_table(cur, "species"):
        cols = colnames(cur, "species")
        if "species_key" not in cols and "id" in cols:
            print("→ species: renaming id → species_key")
            cur.execute("ALTER TABLE species RENAME COLUMN id TO species_key")
            conn.commit()

    # 2) species_traits: ensure FK is species_key
    if has_table(cur, "species_traits"):
        cols = colnames(cur, "species_traits")
        if "species_key" not in cols and "species_id" in cols:
            print("→ species_traits: renaming species_id → species_key")
            cur.execute("ALTER TABLE species_traits RENAME COLUMN species_id TO species_key")
            conn.commit()

    # 3) occurrences: ensure FK is species_key
    if has_table(cur, "occurrences"):
        cols = colnames(cur, "occurrences")
        if "species_key" not in cols and "species_id" in cols:
            print("→ occurrences: renaming species_id → species_key")
            cur.execute("ALTER TABLE occurrences RENAME COLUMN species_id TO species_key")
            conn.commit()

    # 4) species_distribution: ensure FK is species_key
    if has_table(cur, "species_distribution"):
        cols = colnames(cur, "species_distribution")
        if "species_key" not in cols and "species_id" in cols:
            print("→ species_distribution: renaming species_id → species_key")
            cur.execute("ALTER TABLE species_distribution RENAME COLUMN species_id TO species_key")
            conn.commit()

    # 5) sanity: add canonical_binomial if missing (handy for joins)
    if has_table(cur, "species"):
        cols = colnames(cur, "species")
        if "canonical_binomial" not in cols:
            print("→ species: adding canonical_binomial TEXT")
            cur.execute("ALTER TABLE species ADD COLUMN canonical_binomial TEXT")
            conn.commit()

    print("✅ Migration complete. All tables now use species_key.")
    conn.close()

if __name__ == "__main__":
    main()

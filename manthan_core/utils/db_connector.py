# manthan_core/utils/db_connector.py
from __future__ import annotations
from typing import List, Dict, Any

from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.engine import Engine

class KnowledgeBase:
    """Connector to the Manthan knowledge base (SQLite or PostgreSQL)."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine: Engine = create_engine(self.db_url, future=True)
        self._ensure_species_key_views()

    # ------- internal helpers -------

    def _table_exists(self, name: str) -> bool:
        sql = text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n")
        try:
            with self.engine.begin() as conn:
                row = conn.execute(sql, {"n": name}).fetchone()
            return row is not None
        except Exception:
            try:
                with self.engine.begin() as conn:
                    row = conn.execute(text("SELECT to_regclass(:n)"), {"n": name}).fetchone()
                return row and list(row)[0] is not None
            except Exception:
                return False

    def _ensure_species_key_views(self) -> None:
        """If a legacy DB uses id/species_id, normalize with compatibility views."""
        with self.engine.begin() as conn:
            # species view
            if self._table_exists("species"):
                cols = [r["name"] for r in conn.execute(text("PRAGMA table_info(species)")).mappings()]
                if "species_key" not in cols and "id" in cols:
                    conn.execute(text("DROP VIEW IF EXISTS species_v"))
                    conn.execute(text(
                        "CREATE VIEW species_v AS SELECT id AS species_key, * FROM species"
                    ))
                else:
                    conn.execute(text("DROP VIEW IF EXISTS species_v"))
                    conn.execute(text("CREATE VIEW species_v AS SELECT * FROM species"))
            # species_traits view
            if self._table_exists("species_traits"):
                cols = [r["name"] for r in conn.execute(text("PRAGMA table_info(species_traits)")).mappings()]
                if "species_key" not in cols and "species_id" in cols:
                    conn.execute(text("DROP VIEW IF EXISTS species_traits_v"))
                    conn.execute(text(
                        "CREATE VIEW species_traits_v AS SELECT species_id AS species_key, * FROM species_traits"
                    ))
                else:
                    conn.execute(text("DROP VIEW IF EXISTS species_traits_v"))
                    conn.execute(text("CREATE VIEW species_traits_v AS SELECT * FROM species_traits"))
            # occurrences view
            if self._table_exists("occurrences"):
                cols = [r["name"] for r in conn.execute(text("PRAGMA table_info(occurrences)")).mappings()]
                if "species_key" not in cols and "species_id" in cols:
                    conn.execute(text("DROP VIEW IF EXISTS occurrences_v"))
                    conn.execute(text(
                        "CREATE VIEW occurrences_v AS SELECT species_id AS species_key, * FROM occurrences"
                    ))
                else:
                    conn.execute(text("DROP VIEW IF EXISTS occurrences_v"))
                    conn.execute(text("CREATE VIEW occurrences_v AS SELECT * FROM occurrences"))
            # species_distribution view
            if self._table_exists("species_distribution"):
                cols = [r["name"] for r in conn.execute(text("PRAGMA table_info(species_distribution)")).mappings()]
                if "species_key" not in cols and "species_id" in cols:
                    conn.execute(text("DROP VIEW IF EXISTS species_distribution_v"))
                    conn.execute(text(
                        "CREATE VIEW species_distribution_v AS SELECT species_id AS species_key, * FROM species_distribution"
                    ))
                else:
                    conn.execute(text("DROP VIEW IF EXISTS species_distribution_v"))
                    conn.execute(text("CREATE VIEW species_distribution_v AS SELECT * FROM species_distribution"))

    # ------- public API -------

    def get_species_traits(self, species_list: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Fetch all traits for the given species list.
        Matches both scientific_name and canonical_name.
        Returns: {<input_species_name>: {trait_name: trait_value, ...}, ...}
        """
        if not species_list:
            return {}

        if not self._table_exists("species") or not self._table_exists("species_traits"):
            return {}

        sql = text(
            """
            SELECT
                s.scientific_name,
                s.canonical_name,
                st.trait_name,
                st.trait_value
            FROM species_v s
            JOIN species_traits_v st
              ON st.species_key = s.species_key
            WHERE s.scientific_name IN :names
               OR s.canonical_name  IN :names
            """
        ).bindparams(bindparam("names", expanding=True))

        with self.engine.begin() as conn:
            rows = conn.execute(sql, {"names": list(species_list)}).mappings().all()

        by_key: Dict[str, Dict[str, str]] = {}
        for r in rows:
            sci = (r.get("scientific_name") or "").strip()
            can = (r.get("canonical_name") or "").strip()
            tname = (r.get("trait_name") or "").strip()
            tval = (r.get("trait_value") or "").strip()
            if tname:
                if sci:
                    by_key.setdefault(sci, {})[tname] = tval
                if can:
                    by_key.setdefault(can, {})[tname] = tval

        result: Dict[str, Dict[str, str]] = {}
        lower_index = {k.lower(): v for k, v in by_key.items()}
        for requested in species_list:
            v = by_key.get(requested)
            if v is None:
                v = lower_index.get(requested.lower(), {})
            result[requested] = v

        return result

    def get_economic_data(self, species_list: List[str]) -> Dict[str, Any]:
        """
        Optional table:
          economic_data(species_key INTEGER, product_type TEXT, yield_per_ha REAL, price_per_unit REAL)
        Returns: {scientific_name: {'type': ..., 'yield_per_ha': ..., 'price': ...}}
        """
        if not species_list:
            return {}

        if not self._table_exists("species") or not self._table_exists("economic_data"):
            return {}

        sql = text(
            """
            SELECT
                s.scientific_name,
                ed.product_type,
                ed.yield_per_ha,
                ed.price_per_unit
            FROM species_v s
            JOIN economic_data ed
              ON ed.species_key = s.species_key
            WHERE s.scientific_name IN :names
               OR s.canonical_name  IN :names
            """
        ).bindparams(bindparam("names", expanding=True))

        with self.engine.begin() as conn:
            rows = conn.execute(sql, {"names": list(species_list)}).mappings().all()

        out: Dict[str, Any] = {}
        for r in rows:
            nm = r.get("scientific_name")
            if not nm:
                continue
            out[nm] = {
                "type": r.get("product_type"),
                "yield_per_ha": r.get("yield_per_ha"),
                "price": r.get("price_per_unit"),
            }
        return out

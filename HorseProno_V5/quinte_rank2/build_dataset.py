"""Transforme le cache PMU en deux tables au format de validated_history.csv (ClaudeV5).

- history.csv : toutes les courses collectées (France + quintés). Comme dans ClaudeV5,
  finish_position n'est renseignée que pour les places 1..5 (le reste NaN).
- quintes.csv : une ligne par quinté (≥ quinte_from) avec heure de départ et rapports
  officiels SIMPLE_GAGNANT / SIMPLE_PLACE (euros pour 1 €), ou NaN si absents.

Cote : `dernierRapportReference.rapport`, conservée SEULEMENT si sa dateRapport est
strictement antérieure à l'heure de départ ; sinon NaN (odds_missing dans le modèle).
`dernierRapportDirect` (cote finale, publiée après le départ) n'est jamais utilisée.
"""
from __future__ import annotations
import argparse, gzip, json, os
from datetime import datetime
from pathlib import Path
import pandas as pd

CACHE = Path(os.environ.get("PMU_CACHE", "/tmp/pmu_cache"))


def load(p: Path):
    return json.loads(gzip.decompress(p.read_bytes())) if p.exists() else None


def discipline(c) -> str:
    d, cat = c.get("discipline", ""), str(c.get("categorieParticularite", ""))
    if d == "ATTELE":
        return "ATTELE_AUTOSTART" if "AUTOSTART" in cat else "ATTELE_VOLTE"
    return {"MONTE": "TROT_MONTE", "HAIE": "HAIES", "CROSS": "CROSS_COUNTRY"}.get(d, d)


def payouts(rap, kind):
    """{numéro: euros pour 1 €} pour SIMPLE_GAGNANT / SIMPLE_PLACE (repli e-version)."""
    if not rap:
        return None, None
    by = {x["typePari"]: x for x in rap}
    x = by.get(kind) or by.get("E_" + kind)
    if not x:
        return None, None
    out = {}
    for r in x.get("rapports", []):
        for n in str(r.get("combinaison", "")).split("-"):
            if n.strip().isdigit():
                out[int(n)] = r.get("dividendePourUnEuro", 0) / 100.0
    return out, x["typePari"]


def main(quinte_from: str, out_dir: Path):
    rows, qrows = [], []
    for day in sorted(CACHE.iterdir()):
        prog = load(day / "programme.json.gz")
        if not prog:
            continue
        dt = datetime.strptime(day.name, "%d%m%Y").date()
        for r in prog["programme"]["reunions"]:
            for c in r["courses"]:
                rc = f"R{r['numOfficiel']}C{c['numOrdre']}"
                parts = load(day / rc / "participants.json.gz")
                if not parts:
                    continue
                race_id = f"{rc}_{dt}"
                start = c.get("heureDepart")
                quinte = any(p.get("typePari") == "QUINTE_PLUS" for p in c.get("paris", []))
                for p in parts.get("participants", []):
                    ref = p.get("dernierRapportReference") or {}
                    odds = ref.get("rapport") if (ref.get("dateRapport") and start and ref["dateRapport"] < start) else None
                    pos = p.get("ordreArrivee")
                    pw = p.get("poidsConditionMonte")
                    rows.append({
                        "race_id": race_id, "race_date": str(dt), "meeting_number": r["numOfficiel"],
                        "race_number": c["numOrdre"], "is_quinte": quinte, "start_ms": start,
                        "discipline": discipline(c), "hippodrome": r["hippodrome"]["libelleCourt"],
                        "distance": c.get("distance"), "terrain": "INCONNU",
                        "horse_number": p["numPmu"], "horse_name": p.get("nom"),
                        "jockey": p.get("driver"), "trainer": p.get("entraineur"),
                        "odds": odds, "odds_ref_ms": ref.get("dateRapport"),
                        "draw": p.get("placeCorde"), "weight": pw / 10 if pw else None,
                        "age": p.get("age"), "sex": p.get("sexe"), "recent_form": p.get("musique", ""),
                        "finish_position": pos if isinstance(pos, int) and 1 <= pos <= 5 else None,
                        "finish_full": pos if isinstance(pos, int) else None,
                        "is_non_runner": p.get("statut") != "PARTANT",
                    })
                if quinte and str(dt) >= quinte_from:
                    rap = load(day / rc / "rapports.json.gz") or load(day / rc / "rapports_internet.json.gz")
                    g, gt = payouts(rap, "SIMPLE_GAGNANT")
                    pl, pt = payouts(rap, "SIMPLE_PLACE")
                    qrows.append({"race_id": race_id, "race_date": str(dt), "start_ms": start,
                                  "hippodrome": r["hippodrome"]["libelleCourt"], "discipline": discipline(c),
                                  "gagnant": json.dumps(g) if g is not None else None, "gagnant_type": gt,
                                  "place": json.dumps(pl) if pl is not None else None, "place_type": pt})
    h = pd.DataFrame(rows)
    h["field_size"] = h.groupby("race_id").horse_number.transform("size")
    out_dir.mkdir(parents=True, exist_ok=True)
    h.to_csv(out_dir / "history.csv", index=False)
    pd.DataFrame(qrows).to_csv(out_dir / "quintes.csv", index=False)
    print(len(h), h.race_id.nunique(), "courses ;", len(qrows), "quintés")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quinte-from", default="2025-11-01")
    ap.add_argument("--out", default=os.environ.get("PMU_CACHE", "/tmp/pmu_cache") + "_tables")
    a = ap.parse_args()
    main(a.quinte_from, Path(a.out))

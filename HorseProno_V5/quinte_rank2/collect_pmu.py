"""Collecte PMU (API turfinfo non officielle), 1 requête/s max, reprenable (cache disque).

Pour chaque jour : programme ; pour chaque course française (et tout quinté) : participants
(INTERNET) ; pour chaque quinté : rapports définitifs (officiels, sans spécialisation,
repli INTERNET). Le cache brut est écrit dans CACHE (hors dépôt).
"""
from __future__ import annotations
import argparse, gzip, json, os, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

API = "https://online.turfinfo.api.pmu.fr/rest/client/1/programme"
CACHE = Path(os.environ.get("PMU_CACHE", "/tmp/pmu_cache"))
_last = [0.0]
_lock = threading.Lock()


def _throttle():
    """Espace les DÉPARTS de requêtes d'au moins 1,05 s, quel que soit le nombre de threads."""
    with _lock:
        wait = 1.05 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def get(url: str, path: Path):
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    for attempt in range(5):
        _throttle()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-client"}), timeout=30) as r:
                data = json.loads(r.read() or b"null")
            break
        except urllib.error.HTTPError as e:
            if e.code in (204, 404):
                data = None
                break
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    else:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    return data


def is_quinte(c):
    return any(p.get("typePari") == "QUINTE_PLUS" for p in c.get("paris", []))


def fetch_race(ds, rc, want_rapports):
    folder = CACHE / ds / rc.replace("/", "")
    get(f"{API}/{ds}/{rc}/participants?specialisation=INTERNET", folder / "participants.json.gz")
    if want_rapports:
        if not get(f"{API}/{ds}/{rc}/rapports-definitifs", folder / "rapports.json.gz"):
            get(f"{API}/{ds}/{rc}/rapports-definitifs?specialisation=INTERNET", folder / "rapports_internet.json.gz")


def run(start: date, end: date, quinte_from: date):
    d = start
    while d <= end:
        ds = d.strftime("%d%m%Y")
        prog = get(f"{API}/{ds}", CACHE / ds / "programme.json.gz")
        jobs = []
        for r in ((prog or {}).get("programme") or {}).get("reunions", []):
            fra = (r.get("pays") or {}).get("code") == "FRA"
            for c in r.get("courses", []):
                q = is_quinte(c)
                if fra or q:
                    jobs.append((f"R{r['numOfficiel']}/C{c['numOrdre']}", q and d >= quinte_from))
        with ThreadPoolExecutor(3) as ex:
            list(ex.map(lambda j: fetch_race(ds, *j), jobs))
        n = len(jobs)
        print(d, n, flush=True)
        d += timedelta(days=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-09-01")
    ap.add_argument("--end", default="2026-11-09")
    ap.add_argument("--quinte-from", default="2025-11-01")
    a = ap.parse_args()
    run(date.fromisoformat(a.start), date.fromisoformat(a.end), date.fromisoformat(a.quinte_from))

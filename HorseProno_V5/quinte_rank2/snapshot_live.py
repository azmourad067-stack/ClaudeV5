"""Relève les participants d'une course à des instants donnés (avant / après départ) pour
mesurer quels champs de l'API changent après l'arrivée. Quelques requêtes seulement."""
import gzip, json, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

date, rc, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
times = sys.argv[4:]  # heures UTC HH:MM
url = f"https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{date}/{rc}/participants?specialisation=INTERNET"
out.mkdir(parents=True, exist_ok=True)
for t in times:
    now = datetime.now(timezone.utc)
    target = now.replace(hour=int(t[:2]), minute=int(t[3:]), second=0, microsecond=0)
    if target > now:
        time.sleep((target - now).total_seconds())
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-client"}), timeout=30) as r:
        data = r.read()
    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    (out / f"snap_{stamp}.json.gz").write_bytes(gzip.compress(data))
    print("snapshot", stamp, flush=True)

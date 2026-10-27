"""HorseProno V5 — modèle probabiliste unifié (logit conditionnel « exploded »).

Principe (approche Benter / Plackett-Luce, standard en modélisation hippique) :
  1. un score linéaire u_i par partant, construit uniquement sur des informations
     pré-course (marché, musique PMU, historique interne cheval/jockey/entraîneur,
     poids, corde, âge…) ;
  2. P(gagner) = softmax(u) dans la course ; l'arrivée complète suit un modèle
     de Plackett-Luce (on retire le gagnant et on recommence) ;
  3. les coefficients sont appris par maximum de vraisemblance sur les arrivées
     1..K de TOUTES les courses de l'historique (≈5 000 courses), pas seulement
     sur les ~100 Quintés ;
  4. P(Top5) de chaque cheval est obtenue par simulation Plackett-Luce ;
     la shortlist Quinté = 7 meilleures P(Top5), l'ordre = P(gagner).

Aucun numpy/scipy exotique : numpy + scipy.optimize (L-BFGS) uniquement à
l'entraînement ; l'inférence ne demande que numpy/pandas.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quinte_candidate import key, prepare

# ---------------------------------------------------------------------------
# Musique PMU (pré-course : vérifié — le 1er symbole ne coïncide avec l'arrivée
# du jour que dans 12 % des cas, contre 84 % pour la musique de la course suivante)
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"(\(\d{2}\))|([0-9]|[A-Za-z]{1,3}?)([apmhsc])", re.I)
_POS_SCORE = {1: 1.0, 2: 0.80, 3: 0.65, 4: 0.52, 5: 0.42, 6: 0.34, 7: 0.28, 8: 0.23, 9: 0.18, 0: 0.08}
_DISC_LETTER = {
    "PLAT": "p", "ATTELE_AUTOSTART": "a", "ATTELE_VOLTE": "a", "TROT_MONTE": "m",
    "HAIES": "h", "STEEPLECHASE": "s", "CROSS_COUNTRY": "c",
}
MUSIC_COLS = ["mu_n", "mu_last", "mu_mean3", "mu_mean6", "mu_win6", "mu_top3_6", "mu_fail6",
              "mu_same_disc6", "mu_layoff", "mu_missing"]


def parse_music(value: Any, discipline: str = "") -> dict[str, float]:
    s = str(value or "").strip()
    runs: list[tuple[float, bool, str]] = []  # (score, fail, letter)
    layoff = 0.0
    for m in _TOKEN.finditer(s):
        if m.group(1):
            if not runs:
                layoff = 1.0  # aucune sortie depuis le début de l'année : longue absence
            continue
        tok, letter = m.group(2), m.group(3).lower()
        if tok.isdigit():
            runs.append((_POS_SCORE[int(tok)], False, letter))
        else:
            runs.append((0.0, True, letter))  # D(isq.), A(rrêté), T(ombé), Ret…
        if len(runs) >= 10:
            break
    if not runs:
        return {c: 0.0 for c in MUSIC_COLS} | {"mu_missing": 1.0}
    sc = np.array([r[0] for r in runs])
    fail = np.array([r[1] for r in runs], dtype=float)
    six = slice(0, 6)
    target = _DISC_LETTER.get(str(discipline).upper(), "")
    win = np.array([r[0] == 1.0 for r in runs], dtype=float)
    top3 = np.array([r[0] >= 0.65 for r in runs], dtype=float)
    same = np.array([r[2] == target for r in runs], dtype=float)
    return {
        "mu_n": min(len(runs), 10) / 10.0,
        "mu_last": float(sc[0]),
        "mu_mean3": float(sc[:3].mean()),
        "mu_mean6": float(sc[six].mean()),
        "mu_win6": float(win[six].mean()),
        "mu_top3_6": float(top3[six].mean()),
        "mu_fail6": float(fail[six].mean()),
        "mu_same_disc6": float(same[six].mean()) if target else 0.0,
        "mu_layoff": layoff,
        "mu_missing": 0.0,
    }


def music_frame(race: pd.DataFrame) -> pd.DataFrame:
    rec = [parse_music(m, d) for m, d in zip(race["recent_form"].fillna(""), race["discipline"].fillna(""))]
    return pd.DataFrame(rec, index=race.index)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
TROT = {"ATTELE_AUTOSTART", "ATTELE_VOLTE", "TROT_MONTE"}
OBST = {"HAIES", "STEEPLECHASE", "CROSS_COUNTRY"}


def relevant_history(history: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
    """Ne garde que les courses (complètes) où apparaît un cheval, jockey ou entraîneur
    de la course cible. prepare() ne lit que ces lignes : résultat identique
    (vérifié numériquement), calcul ~10x plus rapide."""
    mask = pd.Series(False, index=history.index)
    for col in ["horse_name", "jockey", "trainer"]:
        wanted = set(key(race[col])) - {""}
        mask |= key(history[col]).isin(wanted)
    return history[history["race_id"].isin(history.loc[mask, "race_id"].unique())]


def build_features(history: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    """Features pré-course d'une ou plusieurs courses. `history` ne doit contenir
    que des courses terminées ; prepare() n'utilise que les dates strictement
    antérieures à chaque course de `query`."""
    q = query.reset_index(drop=True).copy()
    base = prepare(history, q)
    mu = music_frame(q)
    gid = q["race_id"].astype(str)
    disc = q["discipline"].fillna("").str.upper()
    is_trot = disc.isin(TROT).astype(float)
    is_galop = 1.0 - is_trot
    is_obst = disc.isin(OBST).astype(float)
    odds_ok = 1.0 - base["odds_missing"]

    f = pd.DataFrame(index=q.index)
    # Marché : log-probabilité implicite normalisée (≈ log-cote, dé-marginée)
    f["log_mkt"] = np.log(base["market_prob"].clip(1e-4, 1.0))
    f["log_mkt_trot"] = f["log_mkt"] * is_trot
    f["odds_missing"] = base["odds_missing"]
    # Musique (relatif course : centré)
    for c in MUSIC_COLS:
        f[c] = mu[c]
    f["mu_fail6_trot"] = mu["mu_fail6"] * is_trot
    f["mu_same_disc6_obst"] = mu["mu_same_disc6"] * is_obst
    # Historique interne (shrinké dans prepare)
    for c in ["horse_name_top5", "horse_name_win", "jockey_top5", "jockey_win", "trainer_top5",
              "trainer_win", "horse_name_starts", "jockey_starts", "trainer_starts", "form", "consistency"]:
        f[c] = base[c]
    # Physique de course
    f["weight_galop"] = base["weight"] * is_galop * (1 - base["weight_missing"])
    f["draw_plat"] = base["plat_draw"] * (1 - base["draw_missing"])
    f["draw_plat_sq"] = (base["plat_draw"] ** 2) * (1 - base["draw_missing"])
    f["draw_autostart"] = base["autostart_draw"] * (1 - base["draw_missing"])
    age = base["age"]
    f["age_young"] = (age <= 3).astype(float)
    f["age_old"] = ((age >= 9) & (is_trot == 0)).astype(float) + ((age >= 11) & (is_trot == 1)).astype(float)
    f["sex_FEMELLES"] = base["sex_FEMELLES"]
    # Centrage intra-course (le logit conditionnel est invariant aux constantes de course,
    # mais le centrage stabilise l'optimisation et rend les coefficients lisibles)
    f = f - f.groupby(gid).transform("mean")
    f["odds_ok"] = odds_ok  # non utilisé comme feature, utile au diagnostic
    return f


# Jeu retenu en production : marché + historique interne + musique.
# Les variables physiques (poids, corde, âge, sexe) ont été testées : gain nul sur la
# validation 01/08→11/09 (Top5-dans-7 : 3,917 vs 3,915) → écartées par parcimonie.
FEATURES_PROD = [
    "log_mkt", "log_mkt_trot", "odds_missing",
    "horse_name_top5", "horse_name_win", "jockey_top5", "jockey_win", "trainer_top5", "trainer_win",
    "horse_name_starts", "jockey_starts", "trainer_starts", "form", "consistency",
    "mu_n", "mu_last", "mu_mean3", "mu_mean6", "mu_win6", "mu_top3_6", "mu_fail6", "mu_fail6_trot",
    "mu_same_disc6", "mu_same_disc6_obst", "mu_layoff", "mu_missing",
]

FEATURES_V5 = [
    "log_mkt", "log_mkt_trot", "odds_missing",
    "mu_n", "mu_last", "mu_mean3", "mu_mean6", "mu_win6", "mu_top3_6", "mu_fail6", "mu_fail6_trot",
    "mu_same_disc6", "mu_same_disc6_obst", "mu_layoff", "mu_missing",
    "horse_name_top5", "horse_name_win", "jockey_top5", "jockey_win", "trainer_top5", "trainer_win",
    "horse_name_starts", "jockey_starts", "trainer_starts", "form", "consistency",
    "weight_galop", "draw_plat", "draw_plat_sq", "draw_autostart", "age_young", "age_old", "sex_FEMELLES",
]


# ---------------------------------------------------------------------------
# Logit conditionnel exploded (Plackett-Luce) — entraînement
# ---------------------------------------------------------------------------
def _stage_data(X: np.ndarray, groups: np.ndarray, pos: np.ndarray, k: int):
    """Pré-calcule, pour chaque étape (course, rang r<=k), l'ensemble encore en lice."""
    stages = []  # (idx_chosen, idx_remaining_array, stage_rank)
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    bounds = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        p = pos[idx]
        remaining = idx.copy()
        for r in range(1, k + 1):
            win = idx[p == r]
            if len(win) != 1:
                break
            stages.append((int(win[0]), remaining.copy(), r))
            remaining = remaining[remaining != win[0]]
            if len(remaining) < 2:
                break
    return stages


def fit_exploded_logit(X: np.ndarray, groups: np.ndarray, pos: np.ndarray, k: int = 5,
                       l2: float = 1.0, stage_weights: list[float] | None = None) -> np.ndarray:
    from scipy.optimize import minimize

    stages = _stage_data(X, groups, pos, k)
    sw = stage_weights or [1.0] * k
    # aplatissement vectorisé
    chosen = np.array([s[0] for s in stages])
    lens = np.array([len(s[1]) for s in stages])
    flat = np.concatenate([s[1] for s in stages])
    seg = np.repeat(np.arange(len(stages)), lens)
    w = np.array([sw[s[2] - 1] for s in stages])
    Xf = X[flat]
    Xc = X[chosen]
    n_obs = w.sum()

    def obj(beta):
        u = Xf @ beta
        # log-sum-exp par segment
        m = np.full(len(stages), -np.inf)
        np.maximum.at(m, seg, u)
        e = np.exp(u - m[seg])
        s = np.bincount(seg, weights=e, minlength=len(stages))
        lse = m + np.log(s)
        ll = (w * (Xc @ beta - lse)).sum()
        p = e / s[seg]
        grad_ll = (w[:, None] * Xc).sum(0) - (Xf * (p * w[seg])[:, None]).sum(0)
        return -(ll / n_obs) + l2 * (beta @ beta) / n_obs, -(grad_ll / n_obs) + 2 * l2 * beta / n_obs

    beta0 = np.zeros(X.shape[1])
    res = minimize(obj, beta0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x


# ---------------------------------------------------------------------------
# Inférence
# ---------------------------------------------------------------------------
def softmax_by_race(u: np.ndarray, groups: np.ndarray) -> np.ndarray:
    s = pd.Series(u)
    m = s.groupby(groups).transform("max").to_numpy()
    e = np.exp(u - m)
    return e / pd.Series(e).groupby(groups).transform("sum").to_numpy()


def plackett_luce_topk(strength: np.ndarray, k: int = 5, n_sim: int = 20000, seed: int = 7) -> np.ndarray:
    """P(cheval dans les k premiers) par simulation Plackett-Luce (Gumbel-max)."""
    rng = np.random.default_rng(seed)
    logw = np.log(np.clip(strength, 1e-12, None))
    g = rng.gumbel(size=(n_sim, len(strength))) + logw
    topk = np.argpartition(-g, kth=min(k, len(strength) - 1), axis=1)[:, :k]
    counts = np.bincount(topk.ravel(), minlength=len(strength))
    return counts / n_sim


class V5Model:
    """Deux têtes partageant les mêmes features :
    - `win`  : logit conditionnel sur le gagnant (K=1) -> P(gagner), ordre du Top 7 ;
    - `top5` : logit « exploded » sur les places 1..5 (K=5) -> forces Plackett-Luce,
               P(Top5) simulée -> composition du Top 7.
    `win_temperature` (>1 = moins confiant) corrige la surconfiance observée quand le
    modèle, appris sur l'historique, est appliqué à des cotes lues ~T-20 (plus bruitées).
    Elle ne change aucun classement, seulement les probabilités affichées.
    """

    def __init__(self, artifact: dict[str, Any]):
        self.art = artifact
        self.features = artifact["features"]
        self.mean = np.asarray(artifact["mean"], float)
        self.scale = np.asarray(artifact["scale"], float)
        self.coef_win = np.asarray(artifact["coef_win"], float)
        self.coef_top5 = np.asarray(artifact["coef_top5"], float)
        self.temperature = float(artifact.get("win_temperature", 1.0))

    @classmethod
    def load(cls, path: str | Path) -> "V5Model":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _z(self, feats: pd.DataFrame) -> np.ndarray:
        return (feats[self.features].to_numpy(float) - self.mean) / self.scale

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame, n_sim: int = 20000) -> pd.DataFrame:
        race = race.reset_index(drop=True).copy()
        if race["race_id"].nunique() != 1:
            raise ValueError("predict_race attend une seule course")
        feats = build_features(relevant_history(history, race), race)
        z = self._z(feats)
        gid = race["race_id"].astype(str).to_numpy()
        p_win = softmax_by_race((z @ self.coef_win) / self.temperature, gid)
        strength5 = softmax_by_race(z @ self.coef_top5, gid)
        n = len(race)
        p_top5 = plackett_luce_topk(strength5, k=min(5, n), n_sim=n_sim)
        p_top3 = plackett_luce_topk(strength5, k=min(3, n), n_sim=n_sim, seed=11)

        out = race.copy()
        out["v5_win_prob"] = p_win
        out["v5_top3_prob"] = p_top3
        out["v5_top5_prob"] = p_top5
        out["v5_strength5"] = strength5
        odds = pd.to_numeric(out.get("odds"), errors="coerce")
        inv = 1.0 / odds.where(odds > 1)
        out["market_probability"] = (inv / inv.sum()).fillna(0.0)
        out["market_rank"] = out["market_probability"].rank(method="first", ascending=False).astype(int)
        out["v5_fair_odds"] = 1.0 / out["v5_win_prob"].clip(1e-6)
        out["v5_value_ratio"] = (out["v5_win_prob"] * odds).where(odds > 1)
        # Lecture de la musique (affichage)
        mu = music_frame(race)
        out["musique_score6"] = mu["mu_mean6"].round(3)
        out["musique_fail6"] = mu["mu_fail6"].round(3)

        top7 = out.sort_values(["v5_top5_prob", "v5_win_prob", "horse_number"],
                               ascending=[False, False, True]).head(7).index
        out["selected_top7"] = out.index.isin(top7)
        sel = out[out.selected_top7].sort_values(["v5_win_prob", "v5_top5_prob", "horse_number"],
                                                 ascending=[False, False, True])
        rest = out[~out.selected_top7].sort_values(["v5_top5_prob", "v5_win_prob", "horse_number"],
                                                   ascending=[False, False, True])
        out = pd.concat([sel, rest]).reset_index(drop=True)
        out["quinte_rank"] = np.arange(1, n + 1)
        out["selected_top7"] = out["quinte_rank"].le(7)
        out["shortlist_role"] = np.where(out["selected_top7"], "V5_TOP7", "HORS_TOP7")
        out["model_name"] = self.art.get("model_name", "HorseProno_V5")
        return out

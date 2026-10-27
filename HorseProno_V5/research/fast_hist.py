"""Réduit l'historique aux courses qui contiennent au moins un cheval/jockey/entraîneur
de la course cible. prepare() ne dépend que des lignes de ces entités (+ taille de leur
course), donc le résultat est identique — vérifié dans check_fast_hist()."""
from __future__ import annotations
import pandas as pd
from quinte_candidate import key


class FastHistory:
    def __init__(self, h: pd.DataFrame):
        self.h = h.reset_index(drop=True)
        self.idx = {}
        for col in ["horse_name", "jockey", "trainer"]:
            k = key(self.h[col])
            self.idx[col] = self.h.groupby(k).race_id.unique().to_dict()

    def subset(self, race: pd.DataFrame) -> pd.DataFrame:
        rids = set()
        for col in ["horse_name", "jockey", "trainer"]:
            for v in key(race[col]):
                if v and v in self.idx[col]:
                    rids.update(self.idx[col][v])
        d = race.race_date.iloc[0]
        sub = self.h[self.h.race_id.isin(rids)]
        return sub[sub.race_date < d]

"""Tiny pure-Python reader for the numeric LightGBM text model used by HorseProno V3.

This is intentionally narrow: it supports the numeric regression/ranking trees emitted by
our bundled ``quinte_v2_ranker.txt``. It exists so Streamlit Cloud can run the exact
trained ranker even if the optional ``lightgbm`` Python package is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class _Tree:
    split_feature: np.ndarray
    threshold: np.ndarray
    decision_type: np.ndarray
    left_child: np.ndarray
    right_child: np.ndarray
    leaf_value: np.ndarray

    def predict_row(self, row: np.ndarray) -> float:
        node = 0
        while node >= 0:
            fidx = int(self.split_feature[node])
            value = float(row[fidx])
            decision = int(self.decision_type[node])
            # LightGBM's text format encodes default_left in bit 1.  The bundled
            # model uses only numeric <= splits (decision_type=2).
            if np.isnan(value):
                go_left = bool(decision & 2)
            else:
                go_left = value <= float(self.threshold[node])
            node = int(self.left_child[node] if go_left else self.right_child[node])
        leaf_idx = -node - 1
        return float(self.leaf_value[leaf_idx])


class PortableBooster:
    """Drop-in subset of ``lightgbm.Booster`` required by HorseProno V3."""

    def __init__(self, model_file: str | Path):
        self.model_file = str(model_file)
        text = Path(model_file).read_text(encoding="utf-8")
        self.feature_names: list[str] = []
        self.trees: list[_Tree] = []
        self._parse(text)

    @staticmethod
    def _ints(value: str) -> np.ndarray:
        return np.fromstring(value, sep=" ", dtype=np.int64)

    @staticmethod
    def _floats(value: str) -> np.ndarray:
        return np.fromstring(value, sep=" ", dtype=np.float64)

    def _parse(self, text: str) -> None:
        lines = text.splitlines()
        for line in lines:
            if line.startswith("feature_names="):
                self.feature_names = line.split("=", 1)[1].split()
                break

        current: dict[str, str] | None = None
        tree_dicts: list[dict[str, str]] = []
        for line in lines:
            if line.startswith("Tree="):
                if current is not None:
                    tree_dicts.append(current)
                current = {}
                continue
            if current is None:
                continue
            if not line.strip():
                if current:
                    tree_dicts.append(current)
                    current = None
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                current[k] = v
        if current:
            tree_dicts.append(current)

        required = {"split_feature", "threshold", "decision_type", "left_child", "right_child", "leaf_value"}
        for d in tree_dicts:
            if not required.issubset(d):
                continue
            self.trees.append(
                _Tree(
                    split_feature=self._ints(d["split_feature"]),
                    threshold=self._floats(d["threshold"]),
                    decision_type=self._ints(d["decision_type"]),
                    left_child=self._ints(d["left_child"]),
                    right_child=self._ints(d["right_child"]),
                    leaf_value=self._floats(d["leaf_value"]),
                )
            )

        if not self.trees:
            raise ValueError("Aucun arbre LightGBM lisible dans le fichier modèle.")

    def predict(self, data: Any) -> np.ndarray:
        # DataFrame: preserve caller's explicit column order. ndarray: use as-is.
        if hasattr(data, "to_numpy"):
            x = data.to_numpy(dtype=float)
        else:
            x = np.asarray(data, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        out = np.zeros(x.shape[0], dtype=float)
        for tree in self.trees:
            out += np.fromiter((tree.predict_row(row) for row in x), dtype=float, count=x.shape[0])
        return out

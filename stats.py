"""English statistics used by the attack and the UI: letter frequencies, IC, chi-squared, quadgrams."""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
QUADGRAM_FILE = os.path.join(HERE, "data", "quadgrams.npz")
TEST_CORPUS_FILE = os.path.join(HERE, "data", "test_corpus.txt")

# Standard English letter frequencies (percent), A..Z
ENGLISH_PERCENT = [
    8.167, 1.492, 2.782, 4.253, 12.702, 2.228, 2.015, 6.094, 6.966, 0.153, 0.772, 4.025, 2.406,
    6.749, 7.507, 1.929, 0.095, 5.987, 6.327, 9.056, 2.758, 0.978, 2.360, 0.150, 1.974, 0.074,
]
ENGLISH_FREQ = np.array(ENGLISH_PERCENT, dtype=float) / sum(ENGLISH_PERCENT)
RANDOM_IC = 1 / 26
ENGLISH_IC = float((ENGLISH_FREQ**2).sum())  # about 0.0655


def codes(text: str) -> np.ndarray:
    """A-Z string -> int array 0..25."""
    return np.frombuffer(text.encode("ascii"), dtype=np.uint8).astype(np.int64) - 65


def letter_counts(text: str) -> np.ndarray:
    return np.bincount(codes(text), minlength=26) if text else np.zeros(26, dtype=np.int64)


def letter_freq(text: str) -> np.ndarray:
    c = letter_counts(text).astype(float)
    return c / c.sum() if c.sum() else c


def index_of_coincidence(text: str) -> float:
    c = letter_counts(text).astype(float)
    n = c.sum()
    if n < 2:
        return 0.0
    return float((c * (c - 1)).sum() / (n * (n - 1)))


def chi_squared(text: str, expected: np.ndarray = ENGLISH_FREQ) -> float:
    c = letter_counts(text).astype(float)
    n = c.sum()
    if n == 0:
        return 0.0
    e = expected * n
    return float(((c - e) ** 2 / np.where(e == 0, 1, e)).sum())


_QG_CACHE: dict = {}


def load_quadgrams(path: str = QUADGRAM_FILE) -> tuple[np.ndarray, float]:
    """Returns (log10 probability table of size 26**4, reference mean score of real English)."""
    if path not in _QG_CACHE:
        d = np.load(path)
        _QG_CACHE[path] = (d["logp"].astype(np.float32), float(d["ref_mean"]))
    return _QG_CACHE[path]


def quadgram_score(text: str, table: np.ndarray | None = None) -> float:
    """Mean log10 probability per quadgram (about -4.5 for English, about -7 for random text)."""
    if table is None:
        table = load_quadgrams()[0]
    c = codes(text)
    if len(c) < 4:
        return -99.0
    idx = c[:-3] * 17576 + c[1:-2] * 676 + c[2:-1] * 26 + c[3:]
    return float(table[idx].mean())

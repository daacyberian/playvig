"""Ciphertext-only (frequency / statistical) attack on PlayVig.

PlayVig(P) = Vigenere_k2( Playfair_k1(P) ).  The attacker knows the algorithm but not the keys
(Kerckhoffs' principle) and sees only the ciphertext. The attack peels the two stages off in reverse:

 1. Key length      - Index of Coincidence of the ciphertext columns for every candidate period.
                      (Playfair output is not flat: IC ~0.050 vs 0.0385 for random letters.)
 2. Vigenere key    - Every column is a shifted copy of the same Playfair-output letter distribution.
                      Cross-correlating the columns gives all shifts relative to each other.
                      The last unknown (one global shift) is fixed by a leak: Playfair has only 25 letters
                      (I/J merged), so the letter J never appears before the Vigenere stage.
                      The same leak is used to repair single columns.
 3. Playfair square - what is left is a pure Playfair ciphertext. It is solved by simulated annealing
                      over the 25! squares, scored with English quadgram statistics.
 4. Polish          - the recovered plaintext is re-encrypted to re-check the Vigenere key.

The attacker never uses the real keys. No dictionary of keys is used either.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

import cipher
import stats

J = 9  # letter J as a number
CLIP = -6.5  # lowest quadgram log-probability used inside the Playfair search


# --------------------------------------------------------------------------
# Stage 1 + 2: Vigenere layer
# --------------------------------------------------------------------------
def column_counts(c: np.ndarray, period: int) -> np.ndarray:
    """counts[j][x] = how often letter x appears at positions j, j+period, j+2*period, ..."""
    return np.array([np.bincount(c[j::period], minlength=26) for j in range(period)], dtype=float)


def ic_by_period(ct: str, max_len: int = 40) -> dict[int, float]:
    """Average IC of the columns for each candidate key length (higher = more likely)."""
    c = stats.codes(ct)
    out: dict[int, float] = {}
    for p in range(2, max_len + 1):
        if len(c) < p * 2:
            break
        cnt = column_counts(c, p)
        n = cnt.sum(axis=1)
        num = (cnt * (cnt - 1)).sum()
        den = (n * (n - 1)).sum()
        out[p] = float(num / den) if den > 0 else 0.0
    return out


def estimate_key_length(ct: str, max_len: int = 40) -> tuple[int, dict[int, float]]:
    """Smallest period whose IC is close to the best one (multiples of the true period all look good)."""
    scores = ic_by_period(ct, max_len)
    if not scores:
        return 1, scores
    best = max(scores.values())
    thr = stats.RANDOM_IC + 0.75 * (best - stats.RANDOM_IC)
    for p in sorted(scores):
        if scores[p] >= thr:
            return p, scores
    return max(scores, key=scores.get), scores


def _align(cols: np.ndarray, rounds: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Find, for every column of letter counts, the shift that lines it up with the others.
    Returns (shifts, summed aligned distribution). Each column is compared with all the others."""
    m = len(cols)
    shifts = np.zeros(m, dtype=int)
    ref = cols[0].copy()
    for j in range(1, m):
        sc = np.array([ref @ np.roll(cols[j], -s) for s in range(26)])
        shifts[j] = int(sc.argmax())
        ref = ref + np.roll(cols[j], -shifts[j])
    for _ in range(rounds):
        changed = False
        for j in range(m):
            own = np.roll(cols[j], -shifts[j])
            ref_wo = ref - own
            sc = np.array([ref_wo @ np.roll(cols[j], -s) for s in range(26)])
            new = int(sc.argmax())
            changed |= new != shifts[j]
            shifts[j] = new
            ref = ref_wo + np.roll(cols[j], -new)
        if not changed:
            break
    return shifts, ref


def recover_vigenere_key(ct: str, period: int) -> tuple[str, dict]:
    """Recover the Vigenere key for a given period (see module docstring). Returns (key, info).

    Even period: Playfair letters at odd and even positions have slightly different statistics, and
    every Vigenere column then sees only one of them. So the columns are aligned in two separate
    groups (even / odd columns) and each group gets its own global shift from the J leak.
    """
    c = stats.codes(ct)
    cnt = column_counts(c, period)
    groups = [list(range(period))] if period % 2 else [list(range(0, period, 2)), list(range(1, period, 2))]
    key = np.zeros(period, dtype=int)
    refs, shifts_all, zero_sets = [], [], []
    for g in groups:
        sh, ref = _align(cnt[g])
        refs.append(ref)
        shifts_all.append(sh)
        zero_sets.append([int(a) for a in np.flatnonzero(ref == ref.min())])
    # candidate global shifts for every group (the aligned letter J+t is the one that never occurs)
    import itertools
    combos = list(itertools.product(*[z[:6] for z in zero_sets]))
    if len(groups) == 2 and len(combos) > 1:
        def rank(cb):
            # a correct key never produces a Playfair digraph with two equal letters
            k = np.zeros(period, dtype=int)
            for gi, g in enumerate(groups):
                for idx, j in enumerate(g):
                    k[j] = (shifts_all[gi][idx] + cb[gi] - J) % 26
            dec = (c - np.resize(k, len(c))) % 26
            eq = int((dec[0 : len(dec) - 1 : 2] == dec[1::2][: len(dec[0 : len(dec) - 1 : 2])]).sum())
            da, db = np.roll(refs[0], -(cb[0] - J)), np.roll(refs[1], -(cb[1] - J))
            return (eq, -float(da @ db))
        combos.sort(key=rank)
    keys = []
    repaired = 0
    for cb in combos[:4]:
        key = np.zeros(period, dtype=int)
        for gi, g in enumerate(groups):
            t = (cb[gi] - J) % 26
            for idx, j in enumerate(g):
                key[j] = (shifts_all[gi][idx] + t) % 26
            # repair columns that contradict the "J never appears" rule
            for idx, j in enumerate(g):
                if cnt[j][(J + key[j]) % 26] > 0:
                    ok = [k for k in range(26) if cnt[j][(J + k) % 26] == 0]
                    if ok:
                        ref_wo = refs[gi] - np.roll(cnt[j], -shifts_all[gi][idx])
                        key[j] = max(ok, key=lambda k: ref_wo @ np.roll(cnt[j], -((k - t) % 26)))
                        repaired += 1
        keys.append("".join(chr(65 + int(k)) for k in key))
    info = {"zero_candidates": max(len(z) for z in zero_sets), "repaired_columns": repaired, "candidates": keys}
    return keys[0], info


def strip_vigenere(ct: str, key: str) -> str:
    """Remove the Vigenere layer. Any J left over (impossible for a correct key) is turned into I."""
    return cipher.vigenere(ct, key, -1).replace("J", "I")


def canonical_period(key: str) -> str:
    """Shortest repeating unit of a key (so 'ABCABC' == 'ABC')."""
    n = len(key)
    for p in range(1, n + 1):
        if n % p == 0 and key[:p] * (n // p) == key:
            return key[:p]
    return key


# --------------------------------------------------------------------------
# Stage 3: Playfair solver (simulated annealing, numpy-vectorised scoring)
# --------------------------------------------------------------------------
SQ_CODES = np.array([ord(ch) - 65 for ch in cipher.SQUARE_ALPHABET], dtype=np.int64)
_POS = np.arange(25)
ROW, COL = _POS // 5, _POS % 5
LEFT = ROW * 5 + (COL - 1) % 5
UP = ((ROW - 1) % 5) * 5 + COL


class PlayfairSolver:
    def __init__(self, ct25: str, table: np.ndarray):
        if len(ct25) % 2:
            ct25 = ct25[:-1]
        c = stats.codes(ct25)
        self.a, self.b = c[0::2], c[1::2]
        self.table = np.maximum(table, CLIP)  # clip so that a few corrupted digraphs do not ruin the score
        self.nquad = max(len(c) - 3, 1)
        self.evals = 0

    def plain_codes(self, sq: np.ndarray) -> np.ndarray:
        inv = np.zeros(26, dtype=np.int64)
        inv[sq] = _POS
        pa, pb = inv[self.a], inv[self.b]
        ca, cb, ra, rb = COL[pa], COL[pb], ROW[pa], ROW[pb]
        row = ra == rb
        col = ca == cb
        qa = np.where(row, LEFT[pa], np.where(col, UP[pa], pa - ca + cb))
        qb = np.where(row, LEFT[pb], np.where(col, UP[pb], pb - cb + ca))
        out = np.empty(len(pa) * 2, dtype=np.int64)
        out[0::2] = sq[qa]
        out[1::2] = sq[qb]
        return out

    def score(self, sq: np.ndarray) -> float:
        """Mean log10 quadgram probability of the plaintext this square would give."""
        self.evals += 1
        p = self.plain_codes(sq)
        idx = p[:-3] * 17576 + p[1:-2] * 676 + p[2:-1] * 26 + p[3:]
        return float(self.table[idx].sum()) / self.nquad

    def anneal(self, rnd: random.Random, iters: int, t_start: float, t_end: float,
               start: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        sq = SQ_CODES.copy() if start is None else start.copy()
        if start is None:
            perm = list(sq)
            rnd.shuffle(perm)
            sq = np.array(perm, dtype=np.int64)
        cur = self.score(sq)
        best_sq, best = sq.copy(), cur
        for it in range(iters):
            temp = t_start + (t_end - t_start) * it / iters
            cand = sq.copy()
            r = rnd.random()
            if r < 0.90:  # swap two letters
                i, j = rnd.sample(range(25), 2)
                cand[i], cand[j] = cand[j], cand[i]
            elif r < 0.92:  # swap two rows
                i, j = rnd.sample(range(5), 2)
                a, b = cand[i * 5 : i * 5 + 5].copy(), cand[j * 5 : j * 5 + 5].copy()
                cand[i * 5 : i * 5 + 5], cand[j * 5 : j * 5 + 5] = b, a
            elif r < 0.94:  # swap two columns
                i, j = rnd.sample(range(5), 2)
                m = cand.reshape(5, 5)
                m[:, [i, j]] = m[:, [j, i]]
            elif r < 0.96:  # flip rows top-bottom
                cand = cand.reshape(5, 5)[::-1].reshape(25).copy()
            elif r < 0.98:  # flip columns left-right
                cand = cand.reshape(5, 5)[:, ::-1].reshape(25).copy()
            else:  # reverse whole square
                cand = cand[::-1].copy()
            s = self.score(cand)
            d = s - cur
            if d >= 0 or rnd.random() < math.exp(d / temp):
                sq, cur = cand, s
                if cur > best:
                    best, best_sq = cur, sq.copy()
        return best_sq, best

    def decrypt(self, sq: np.ndarray) -> str:
        return "".join(chr(65 + int(x)) for x in self.plain_codes(sq))


def square_to_str(sq: np.ndarray) -> str:
    return "".join(chr(65 + int(x)) for x in sq)


# --------------------------------------------------------------------------
# Full attack
# --------------------------------------------------------------------------
@dataclass
class AttackResult:
    key_length: int
    vigenere_key: str
    square: str
    plaintext: str  # Playfair plaintext incl. filler letters
    cleaned: str  # fillers removed (heuristic)
    quad_score: float
    evals: int  # number of quadgram evaluations of the Playfair search (the main effort)
    seconds: float
    stage_seconds: dict = field(default_factory=dict)
    ic_scores: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def playfair_encrypt_raw(prepared: str, square: str) -> str:
    enc, _ = cipher.playfair_tables(square)
    return "".join(enc.get(prepared[i : i + 2], "??") for i in range(0, len(prepared) - 1, 2))


def break_cipher(ct: str, restarts: int = 12, iters: int = 50000, seed: int = 0, max_key_len: int = 40,
                 known_key_length: int | None = None, t_start: float = 0.03, t_end: float = 0.002,
                 progress=None) -> AttackResult:
    """Ciphertext-only attack. progress(stage:str, fraction:float) is optional (used by the UI).

    Effort knobs: `restarts` x `iters` = maximum number of quadgram evaluations of the Playfair search.
    The search stops early as soon as the result looks like English (score >= target).
    """
    t0 = time.perf_counter()
    table, ref_mean = stats.load_quadgrams()
    ct = cipher.normalize(ct)
    if len(ct) % 2:
        ct = ct[:-1]
    rnd = random.Random(seed)
    say = progress or (lambda *_: None)

    # stage 1: key length
    ts = time.perf_counter()
    say("Finding the Vigenere key length (index of coincidence)", 0.05)
    if known_key_length:
        period, ic_scores = known_key_length, {}
    else:
        period, ic_scores = estimate_key_length(ct, max_key_len)
    t_len = time.perf_counter() - ts

    # stage 2: Vigenere key
    ts = time.perf_counter()
    say("Recovering the Vigenere key (column correlation + J leak)", 0.15)
    vkey, vinfo = recover_vigenere_key(ct, period)
    pf_ct = strip_vigenere(ct, vkey)
    t_vig = time.perf_counter() - ts

    # stage 3: Playfair square by simulated annealing, several random restarts.
    # If the Vigenere key is ambiguous (several candidates), the restarts are shared between them.
    ts = time.perf_counter()
    target = ref_mean - 0.40  # "this looks like English" threshold
    evals = 0
    cands = vinfo["candidates"]
    solvers = [PlayfairSolver(strip_vigenere(ct, k), table) for k in cands]
    best_sq, best_score, best_i = None, -1e9, 0
    for r in range(restarts):
        i = r % len(cands)
        say(f"Searching Playfair squares (restart {r + 1}/{restarts})", 0.2 + 0.7 * r / restarts)
        sq, sc = solvers[i].anneal(rnd, iters, t_start, t_end)
        if sc > best_score:
            best_sq, best_score, best_i = sq, sc, i
        if best_score >= target:
            break
    evals += sum(s.evals for s in solvers)
    vkey = cands[best_i]
    solver = solvers[best_i]
    pf_ct = strip_vigenere(ct, vkey)

    # stage 4: polish - re-encrypt the plaintext guess, majority-vote the Vigenere key per column, refine
    polish_rounds = 0
    for _ in range(2):
        plain = solver.decrypt(best_sq)
        pf_est = playfair_encrypt_raw(plain, square_to_str(best_sq))
        est = np.frombuffer(pf_est.encode(), dtype=np.uint8).astype(np.int64) - 65  # '?' -> negative
        ctc = stats.codes(ct[: len(pf_est)])
        valid = est >= 0
        diff = (ctc - est) % 26
        new_key = ""
        for j in range(period):
            dj = diff[j::period][valid[j::period]]
            new_key += chr(65 + int(np.bincount(dj, minlength=26).argmax())) if len(dj) else vkey[j]
        if new_key == vkey:
            break
        vkey = new_key
        polish_rounds += 1
        pf_ct = strip_vigenere(ct, vkey)
        solver = PlayfairSolver(pf_ct, table)
        best_sq, best_score = solver.anneal(rnd, max(iters // 5, 1000), t_end * 3, t_end / 2, start=best_sq)
        evals += solver.evals
    plain = solver.decrypt(best_sq)
    t_pf = time.perf_counter() - ts
    say("Done", 1.0)
    return AttackResult(
        key_length=period,
        vigenere_key=vkey,
        square=square_to_str(best_sq),
        plaintext=plain,
        cleaned=cipher.remove_fillers(plain),
        quad_score=float(best_score),
        evals=evals,
        seconds=time.perf_counter() - t0,
        stage_seconds={"key_length": t_len, "vigenere_key": t_vig, "playfair_search": t_pf},
        ic_scores=ic_scores,
        info={**vinfo, "polish_rounds": polish_rounds, "target_score": target},
    )


def accuracy(recovered: str, truth: str) -> float:
    """Fraction of positions where the recovered plaintext equals the true (padded) plaintext."""
    n = min(len(recovered), len(truth))
    if n == 0:
        return 0.0
    r, t = stats.codes(recovered[:n]), stats.codes(truth[:n])
    return float((r == t).sum() / max(len(truth), 1))

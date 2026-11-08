"""Benchmarks for PlayVig.

 1. speed    : encryption / decryption time of PlayVig vs the Shift cipher for growing input sizes
 2. attack   : success rate and effort of the ciphertext-only attack for several ciphertext lengths
 3. frequency: index of coincidence / chi-squared of plaintext vs ciphertexts (why plain frequency analysis fails)
 4. shift    : how cheaply the Shift cipher falls to the same kind of frequency analysis

Run:  python benchmark.py            (full run, several minutes)
      python benchmark.py --quick    (small run for a smoke test)
Results are written to results/*.csv and results/*.png
"""
from __future__ import annotations

import argparse
import os
import random
import re
import statistics
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import attack
import cipher
import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def corpus_letters() -> str:
    return re.sub(r"[^A-Z]", "", open(stats.TEST_CORPUS_FILE).read().upper())


def random_keys(rnd: random.Random) -> tuple[str, str]:
    pk = "".join(rnd.sample(cipher.SQUARE_ALPHABET, rnd.randint(10, 16)))
    vk = "".join(rnd.choice(cipher.ALPHABET) for _ in range(rnd.randint(10, 16)))
    return pk, vk


# --------------------------------------------------------------------------
# 1. speed
# --------------------------------------------------------------------------
def bench_speed(sizes=(100, 300, 1_000, 10_000, 100_000, 500_000), repeats=3) -> pd.DataFrame:
    text = corpus_letters()
    pk, vk = "SECURITYANALYSTKEY", "NETWORKSECURITY"
    rows = []
    for n in sizes:
        pt = (text * (n // len(text) + 1))[:n]
        ct = cipher.encrypt(pt, pk, vk)
        sct = cipher.shift_encrypt(pt, 7)

        def best(fn):
            ts = []
            for _ in range(repeats):
                t = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t)
            return min(ts)

        rows.append({
            "letters": n,
            "playvig_encrypt_ms": 1000 * best(lambda: cipher.encrypt(pt, pk, vk)),
            "playvig_decrypt_ms": 1000 * best(lambda: cipher.decrypt(ct, pk, vk)),
            "shift_encrypt_ms": 1000 * best(lambda: cipher.shift_encrypt(pt, 7)),
            "shift_decrypt_ms": 1000 * best(lambda: cipher.shift_decrypt(sct, 7)),
        })
    df = pd.DataFrame(rows)
    df["encrypt_ratio"] = df.playvig_encrypt_ms / df.shift_encrypt_ms
    df["decrypt_ratio"] = df.playvig_decrypt_ms / df.shift_decrypt_ms
    return df


# --------------------------------------------------------------------------
# 2. attack
# --------------------------------------------------------------------------
def one_trial(args):
    n, k, restarts, iters = args
    rnd = random.Random(n * 1000 + k)
    text = corpus_letters()
    pk, vk = random_keys(rnd)
    start = rnd.randint(0, len(text) - n - 1)
    st = cipher.encrypt_steps(text[start : start + n], pk, vk)
    r = attack.break_cipher(st["ciphertext"], restarts=restarts, iters=iters, seed=k)
    true_unit = attack.canonical_period(vk)
    len_ok = r.key_length % len(true_unit) == 0
    expanded = (vk * (r.key_length // len(vk) + 1))[: r.key_length] if r.key_length % len(vk) == 0 else None
    vig_ok = expanded is not None and expanded == r.vigenere_key
    acc = attack.accuracy(r.plaintext, st["prepared"])
    return {
        "target_length": n, "trial": k, "ciphertext_letters": len(st["ciphertext"]),
        "true_vigenere_key_len": len(vk), "found_key_len": r.key_length,
        "key_len_ok": len_ok, "vigenere_key_exact": vig_ok, "plaintext_accuracy": acc,
        "broken": acc >= 0.95, "seconds": r.seconds, "quadgram_evals": r.evals,
        "sec_key_length": r.stage_seconds["key_length"], "sec_vigenere": r.stage_seconds["vigenere_key"],
        "sec_playfair": r.stage_seconds["playfair_search"],
    }


def bench_attack(lengths, trials, restarts, iters, workers) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = [(n, k, restarts, iters) for n in lengths for k in range(trials)]
    t = time.time()
    with Pool(workers) as p:
        rows = []
        for i, row in enumerate(p.imap_unordered(one_trial, jobs), 1):
            rows.append(row)
            if i % 10 == 0 or i == len(jobs):
                print(f"  attack trials {i}/{len(jobs)}  ({time.time() - t:.0f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["target_length", "trial"]).reset_index(drop=True)
    g = df.groupby("target_length")
    summary = pd.DataFrame({
        "trials": g.size(),
        "ciphertext_letters_mean": g.ciphertext_letters.mean().round(0),
        "key_length_found_%": (100 * g.key_len_ok.mean()).round(0),
        "vigenere_key_exact_%": (100 * g.vigenere_key_exact.mean()).round(0),
        "plaintext_broken_%": (100 * g.broken.mean()).round(0),
        "mean_letter_accuracy_%": (100 * g.plaintext_accuracy.mean()).round(1),
        "mean_seconds": g.seconds.mean().round(2),
        "median_seconds": g.seconds.median().round(2),
        "mean_quadgram_evals": g.quadgram_evals.mean().round(0),
    }).reset_index()
    return df, summary


# --------------------------------------------------------------------------
# 3. frequency analysis numbers
# --------------------------------------------------------------------------
def bench_frequency(n=2000, trials=30) -> pd.DataFrame:
    text = corpus_letters()
    rnd = random.Random(7)
    acc = {k: [] for k in ("plaintext", "after_playfair", "playvig_ciphertext", "shift_ciphertext")}
    chi = {k: [] for k in acc}
    for _ in range(trials):
        pk, vk = random_keys(rnd)
        s = rnd.randint(0, len(text) - n - 1)
        pt = text[s : s + n]
        st = cipher.encrypt_steps(pt, pk, vk)
        vals = {"plaintext": pt, "after_playfair": st["after_playfair"], "playvig_ciphertext": st["ciphertext"],
                "shift_ciphertext": cipher.shift_encrypt(pt, rnd.randint(1, 25))}
        for k, v in vals.items():
            acc[k].append(stats.index_of_coincidence(v))
            chi[k].append(stats.chi_squared(v))
    rows = [{"text": k, "mean_IC": round(float(np.mean(acc[k])), 4), "mean_chi_squared_vs_english": round(float(np.mean(chi[k])), 0)}
            for k in acc]
    rows.append({"text": "(random letters)", "mean_IC": round(1 / 26, 4), "mean_chi_squared_vs_english": np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. the Shift cipher under the same kind of attack
# --------------------------------------------------------------------------
def shift_break(ct: str) -> tuple[int, int]:
    """Try all 26 keys, keep the one with the lowest chi-squared. Returns (key, tries)."""
    best = min(range(26), key=lambda k: stats.chi_squared(cipher.shift_decrypt(ct, k)))
    return best, 26


def bench_shift(lengths=(20, 50, 100, 200, 500), trials=200) -> pd.DataFrame:
    text = corpus_letters()
    rnd = random.Random(3)
    rows = []
    for n in lengths:
        ok, t_tot = 0, 0.0
        for _ in range(trials):
            key = rnd.randint(1, 25)
            s = rnd.randint(0, len(text) - n - 1)
            ct = cipher.shift_encrypt(text[s : s + n], key)
            t0 = time.perf_counter()
            guess, tries = shift_break(ct)
            t_tot += time.perf_counter() - t0
            ok += guess == key
        rows.append({"letters": n, "trials": trials, "shift_break_success_%": round(100 * ok / trials, 1),
                     "decryptions_tried": 26, "mean_ms": round(1000 * t_tot / trials, 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def make_charts(speed, summary, freq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = summary["ciphertext_letters_mean"]
    ax.plot(x, summary["plaintext_broken_%"], "o-", label="plaintext fully recovered (>=95% letters)", color="#1f5fa8")
    ax.plot(x, summary["vigenere_key_exact_%"], "s--", label="Vigenere key exactly recovered", color="#c0701b")
    ax.plot(x, summary["key_length_found_%"], "^:", label="key length found", color="#3a8a4a")
    ax.set_xlabel("ciphertext length (letters)")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Ciphertext-only attack on PlayVig")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "attack_success.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(x, summary["mean_seconds"], "o-", color="#1f5fa8")
    ax.set_xlabel("ciphertext length (letters)")
    ax.set_ylabel("mean attack time (s)")
    ax.grid(alpha=0.3)
    ax.set_title("Attack effort (2-core sandbox, pure Python/numpy)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "attack_time.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.loglog(speed.letters, speed.playvig_encrypt_ms, "o-", label="PlayVig encrypt")
    ax.loglog(speed.letters, speed.playvig_decrypt_ms, "s-", label="PlayVig decrypt")
    ax.loglog(speed.letters, speed.shift_encrypt_ms, "^--", label="Shift encrypt")
    ax.loglog(speed.letters, speed.shift_decrypt_ms, "v--", label="Shift decrypt")
    ax.set_xlabel("input size (letters)")
    ax.set_ylabel("time (ms)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("Encryption / decryption time (both grow linearly)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "speed.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--lengths", type=int, nargs="*", default=[300, 500, 700, 1000, 1500, 2500])
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--skip-attack", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.trials, a.lengths, a.restarts, a.iters = 2, [500, 1500], 3, 20000
    os.makedirs(OUT, exist_ok=True)

    print("speed benchmark ...", flush=True)
    speed = bench_speed()
    speed.round(3).to_csv(os.path.join(OUT, "speed.csv"), index=False)
    print(speed.round(2).to_string(index=False), flush=True)

    print("frequency numbers ...", flush=True)
    freq = bench_frequency()
    freq.to_csv(os.path.join(OUT, "frequency.csv"), index=False)
    print(freq.to_string(index=False), flush=True)

    print("shift cipher attack ...", flush=True)
    sh = bench_shift()
    sh.to_csv(os.path.join(OUT, "shift_attack.csv"), index=False)
    print(sh.to_string(index=False), flush=True)

    if not a.skip_attack:
        print(f"PlayVig attack: lengths={a.lengths} trials={a.trials} restarts={a.restarts} iters={a.iters}", flush=True)
        df, summary = bench_attack(a.lengths, a.trials, a.restarts, a.iters, a.workers)
        df.to_csv(os.path.join(OUT, "attack_trials.csv"), index=False)
        summary.to_csv(os.path.join(OUT, "attack_summary.csv"), index=False)
        print(summary.to_string(index=False), flush=True)
        make_charts(speed, summary, freq)
    print("done -> results/", flush=True)


if __name__ == "__main__":
    main()

"""Build the English quadgram table used by the attack.

Training books and test books are kept separate on purpose: the attack is trained on
some books and every benchmark plaintext comes from different, held-out books.
Source: NLTK's copy of Project Gutenberg texts (public domain).

Usage: python build_quadgrams.py [path/to/gutenberg.zip]
(if no zip is given it is downloaded from the nltk_data GitHub repo)
"""
import io
import os
import re
import sys
import urllib.request
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip"

TRAIN = [
    "austen-emma", "austen-sense", "chesterton-ball", "chesterton-brown", "edgeworth-parents",
    "melville-moby_dick", "bryant-stories", "burgess-busterbrown",
]
HELDOUT = ["austen-persuasion", "carroll-alice", "chesterton-thursday"]


def letters(text: str) -> str:
    return re.sub(r"[^A-Z]", "", text.upper())


def main():
    if len(sys.argv) > 1:
        data = open(sys.argv[1], "rb").read()
    else:
        data = urllib.request.urlopen(URL, timeout=120).read()
    z = zipfile.ZipFile(io.BytesIO(data))
    read = lambda name: z.read(f"gutenberg/{name}.txt").decode("latin-1")

    train = "".join(letters(read(n)) for n in TRAIN)
    c = np.frombuffer(train.encode(), dtype=np.uint8).astype(np.int64) - 65
    idx = c[:-3] * 17576 + c[1:-2] * 676 + c[2:-1] * 26 + c[3:]
    counts = np.bincount(idx, minlength=26**4).astype(np.float64)
    total = counts.sum()
    floor = np.log10(0.01 / total)  # unseen quadgram
    logp = np.where(counts > 0, np.log10(np.maximum(counts, 1) / total), floor).astype(np.float32)

    # reference score: mean quadgram score of real English that was NOT in the training set
    held = letters(read("austen-persuasion"))[:200000]
    h = np.frombuffer(held.encode(), dtype=np.uint8).astype(np.int64) - 65
    hidx = h[:-3] * 17576 + h[1:-2] * 676 + h[2:-1] * 26 + h[3:]
    ref_mean = float(logp[hidx].mean())

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "data", "quadgrams.npz"), logp=logp, ref_mean=ref_mean)

    # held-out text used for tests / benchmarks / UI samples (readable text, whitespace tidied)
    parts = []
    for n, cap in (("carroll-alice", 140000), ("austen-persuasion", 150000), ("chesterton-thursday", 150000)):
        t = re.sub(r"\s+", " ", read(n))[:cap]
        parts.append(t)
    with open(os.path.join(HERE, "data", "test_corpus.txt"), "w") as f:
        f.write("\n".join(parts))

    print(f"trained on {len(train):,} letters; held-out reference mean = {ref_mean:.3f} log10/quadgram")


if __name__ == "__main__":
    main()

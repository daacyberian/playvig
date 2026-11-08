"""PlayVig: a two-stage classical cipher.

Stage 1 - Playfair (digraph substitution, 5x5 square, keyword >= 10 distinct letters)
Stage 2 - Vigenere (periodic polyalphabetic shift over all 26 letters, key >= 10 letters)

encrypt: plaintext -> normalise -> Playfair -> Vigenere -> ciphertext
decrypt: ciphertext -> Vigenere^-1 -> Playfair^-1 -> plaintext

Also contains a plain Shift cipher (same coding style) used as the efficiency baseline.
"""
from __future__ import annotations

import re

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SQUARE_ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # 25 letters, J merged into I
MIN_KEY_LEN = 10
FILLER = "X"
ALT_FILLER = "Q"  # used only when the letter being padded is itself X


class CipherError(ValueError):
    """Raised for bad keys or bad ciphertext."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Upper-case and keep only A-Z (spaces, digits, punctuation are dropped)."""
    return re.sub(r"[^A-Z]", "", text.upper())


def check_keys(playfair_key: str, vigenere_key: str) -> tuple[str, str]:
    """Validate and normalise both keys. Returns (playfair_key, vigenere_key)."""
    pk = normalize(playfair_key).replace("J", "I")
    vk = normalize(vigenere_key)
    if len(set(pk)) < MIN_KEY_LEN:
        raise CipherError(
            f"Playfair key needs at least {MIN_KEY_LEN} different letters "
            f"(I and J count as one). Got {len(set(pk))}."
        )
    if len(vk) < MIN_KEY_LEN:
        raise CipherError(f"Vigenere key needs at least {MIN_KEY_LEN} letters. Got {len(vk)}.")
    return pk, vk


def build_square(playfair_key: str) -> str:
    """25-letter Playfair square (row by row) from a keyword; J is merged into I."""
    seen: list[str] = []
    for c in (normalize(playfair_key).replace("J", "I") + SQUARE_ALPHABET):
        if c not in seen:
            seen.append(c)
    return "".join(seen)


def square_rows(square: str) -> list[str]:
    return [square[i : i + 5] for i in range(0, 25, 5)]


def _filler_for(letter: str) -> str:
    return FILLER if letter != FILLER else ALT_FILLER


def playfair_prepare(text: str) -> str:
    """A-Z only, J->I, split into digraphs: X inserted between double letters, X pad if odd."""
    s = normalize(text).replace("J", "I")
    out: list[str] = []
    i = 0
    while i < len(s):
        a = s[i]
        if i + 1 < len(s):
            b = s[i + 1]
            if a == b:
                out.append(a + _filler_for(a))
                i += 1
            else:
                out.append(a + b)
                i += 2
        else:
            out.append(a + _filler_for(a))
            i += 1
    return "".join(out)


def remove_fillers(text: str) -> str:
    """Heuristic clean-up of Playfair filler letters (cannot be exact: an original X is ambiguous)."""
    digs = [text[i : i + 2] for i in range(0, len(text), 2)]
    out: list[str] = []
    for j, d in enumerate(digs):
        if len(d) == 2 and d[1] == _filler_for(d[0]):
            if j + 1 < len(digs) and digs[j + 1][0] == d[0]:
                out.append(d[0])
                continue
            if j == len(digs) - 1:
                out.append(d[0])
                continue
        out.append(d)
    return "".join(out)


def playfair_tables(square: str) -> tuple[dict[str, str], dict[str, str]]:
    """Precompute encrypt/decrypt maps for all 600 digraphs (O(1) lookup per digraph)."""
    pos = {c: divmod(i, 5) for i, c in enumerate(square)}
    enc: dict[str, str] = {}
    dec: dict[str, str] = {}
    for a in square:
        for b in square:
            if a == b:
                continue
            (ra, ca), (rb, cb) = pos[a], pos[b]
            if ra == rb:  # same row -> letter to the right
                x = square[ra * 5 + (ca + 1) % 5]
                y = square[rb * 5 + (cb + 1) % 5]
            elif ca == cb:  # same column -> letter below
                x = square[((ra + 1) % 5) * 5 + ca]
                y = square[((rb + 1) % 5) * 5 + cb]
            else:  # rectangle -> swap columns
                x = square[ra * 5 + cb]
                y = square[rb * 5 + ca]
            enc[a + b] = x + y
            dec[x + y] = a + b
    return enc, dec


def vigenere(text: str, key: str, sign: int) -> str:
    """sign=+1 encrypt, sign=-1 decrypt. Works on all 26 letters."""
    k = [ord(c) - 65 for c in key]
    n = len(k)
    return "".join(chr((ord(c) - 65 + sign * k[i % n]) % 26 + 65) for i, c in enumerate(text))


# --------------------------------------------------------------------------
# PlayVig
# --------------------------------------------------------------------------
def encrypt_steps(plaintext: str, playfair_key: str, vigenere_key: str) -> dict:
    """Encrypt and also return every intermediate value (used by the UI)."""
    pk, vk = check_keys(playfair_key, vigenere_key)
    square = build_square(pk)
    enc, _ = playfair_tables(square)
    prepared = playfair_prepare(plaintext)
    if not prepared:
        raise CipherError("Plaintext has no letters A-Z.")
    stage1 = "".join(enc[prepared[i : i + 2]] for i in range(0, len(prepared), 2))
    stage2 = vigenere(stage1, vk, +1)
    return {
        "playfair_key": pk,
        "vigenere_key": vk,
        "square": square,
        "prepared": prepared,
        "after_playfair": stage1,
        "ciphertext": stage2,
    }


def encrypt(plaintext: str, playfair_key: str, vigenere_key: str) -> str:
    return encrypt_steps(plaintext, playfair_key, vigenere_key)["ciphertext"]


def decrypt(ciphertext: str, playfair_key: str, vigenere_key: str, strip_fillers: bool = False) -> str:
    """Returns the padded plaintext (J shown as I). strip_fillers=True removes X fillers heuristically."""
    pk, vk = check_keys(playfair_key, vigenere_key)
    ct = normalize(ciphertext)
    if not ct or len(ct) % 2:
        raise CipherError("Ciphertext must be a non-empty, even number of letters A-Z.")
    square = build_square(pk)
    _, dec = playfair_tables(square)
    stage1 = vigenere(ct, vk, -1)
    out: list[str] = []
    for i in range(0, len(stage1), 2):
        d = stage1[i : i + 2]
        if d not in dec:
            raise CipherError("Wrong key or damaged ciphertext (invalid Playfair digraph).")
        out.append(dec[d])
    text = "".join(out)
    return remove_fillers(text) if strip_fillers else text


# --------------------------------------------------------------------------
# Baseline: Shift cipher (written in the same style so timings are comparable)
# --------------------------------------------------------------------------
def shift_encrypt(plaintext: str, key: int) -> str:
    s = normalize(plaintext)
    return "".join(chr((ord(c) - 65 + key) % 26 + 65) for c in s)


def shift_decrypt(ciphertext: str, key: int) -> str:
    s = normalize(ciphertext)
    return "".join(chr((ord(c) - 65 - key) % 26 + 65) for c in s)


if __name__ == "__main__":
    pk, vk = "SECURITYANALYSTKEY", "NETWORKSECURITY"
    msg = "Meet me at the old library at midnight, bring the documents."
    ct = encrypt(msg, pk, vk)
    print("ciphertext:", ct)
    print("decrypted :", decrypt(ct, pk, vk))
    print("cleaned   :", decrypt(ct, pk, vk, strip_fillers=True))

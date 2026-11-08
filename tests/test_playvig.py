import os
import random
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import attack  # noqa: E402
import cipher  # noqa: E402
import stats  # noqa: E402

PK, VK = "SECURITYANALYSTKEY", "NETWORKSECURITY"


def corpus():
    return re.sub(r"[^A-Z]", "", open(stats.TEST_CORPUS_FILE).read().upper())


@pytest.mark.parametrize("n", [1, 2, 3, 10, 101, 1000, 5000])
def test_roundtrip_many_lengths(n):
    rnd = random.Random(n)
    pt = "".join(rnd.choice(cipher.SQUARE_ALPHABET) for _ in range(n))
    ct = cipher.encrypt(pt, PK, VK)
    assert set(ct) <= set(cipher.ALPHABET) and len(ct) % 2 == 0
    assert cipher.decrypt(ct, PK, VK) == cipher.playfair_prepare(pt)


def test_non_letters_and_case_are_normalised():
    a = cipher.encrypt("Hello, World! 123", PK, VK)
    b = cipher.encrypt("HELLOWORLD", PK, VK)
    assert a == b


def test_filler_removal_simple_text():
    pt = "BALLOONBOX"
    ct = cipher.encrypt(pt, PK, VK)
    assert cipher.decrypt(ct, PK, VK, strip_fillers=True) == pt


def test_j_becomes_i():
    assert cipher.decrypt(cipher.encrypt("JAZZ JUMP", PK, VK), PK, VK, True).startswith("IAZ")


def test_key_rules():
    with pytest.raises(cipher.CipherError):
        cipher.encrypt("hello", "SHORTKEY", VK)
    with pytest.raises(cipher.CipherError):
        cipher.encrypt("hello", PK, "SHORT")
    with pytest.raises(cipher.CipherError):
        cipher.encrypt("12345", PK, VK)


def test_wrong_key_does_not_decrypt():
    pt = "ATTACKATDAWNATTHEEASTGATE"
    ct = cipher.encrypt(pt, PK, VK)
    try:
        out = cipher.decrypt(ct, PK, "WRONGKEYWRONGKEY")
    except cipher.CipherError:
        return
    assert out != cipher.playfair_prepare(pt)


def test_shift_roundtrip():
    assert cipher.shift_decrypt(cipher.shift_encrypt("HELLO WORLD", 5), 5) == "HELLOWORLD"


def test_square_has_25_unique_letters():
    sq = cipher.build_square(PK)
    assert len(sq) == 25 and len(set(sq)) == 25 and "J" not in sq


def test_playfair_output_never_contains_j_or_double_letters():
    rnd = random.Random(1)
    pt = "".join(rnd.choice(cipher.ALPHABET) for _ in range(2000))
    sq = cipher.build_square(PK)
    pf = cipher.encrypt_steps(pt, PK, VK)["after_playfair"]
    assert "J" not in pf
    assert all(pf[i] != pf[i + 1] for i in range(0, len(pf), 2))
    assert len(sq) == 25


def test_attack_finds_key_length_and_vigenere_key_on_long_text():
    text = corpus()
    st_ = cipher.encrypt_steps(text[10000:13000], PK, VK)
    p, _ = attack.estimate_key_length(st_["ciphertext"])
    assert p == len(VK)
    key, _ = attack.recover_vigenere_key(st_["ciphertext"], p)
    assert key == VK


def test_playfair_search_solves_known_stage():
    text = corpus()
    sq = cipher.build_square(PK)
    prepared = cipher.playfair_prepare(text[20000:21200])
    enc, _ = cipher.playfair_tables(sq)
    pf = "".join(enc[prepared[i : i + 2]] for i in range(0, len(prepared), 2))
    table = stats.load_quadgrams()[0]
    solver = attack.PlayfairSolver(pf, table)
    best = -1e9
    for r in range(8):
        s, sc = solver.anneal(random.Random(r), 50000, 0.03, 0.002)
        best = max(best, sc)
        if sc > -4.5:
            assert attack.accuracy(solver.decrypt(s), prepared) > 0.95
            return
    pytest.fail(f"search did not converge, best score {best}")

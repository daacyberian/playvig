# PlayVig - a two-stage classical cipher (Playfair + Vigenere) and an attack on it

Network and Information Security (CT-486), CLO3: design a classical cipher, then break it.

```
plaintext --> [Stage 1: Playfair, 5x5 square, key >= 10 letters] --> [Stage 2: Vigenere, key >= 10 letters] --> ciphertext
```

* Alphabet: A-Z only. Other characters are removed, J is merged into I (standard Playfair rule).
* Playfair padding: `X` is inserted between double letters and at the end of an odd-length text.
* Keys: Playfair key needs 10+ different letters, Vigenere key needs 10+ letters.

## Files

| File | What it does |
| --- | --- |
| `cipher.py` | PlayVig encrypt / decrypt, key checks, Shift cipher baseline |
| `attack.py` | Ciphertext-only attack (key length -> Vigenere key -> Playfair square) |
| `stats.py` | English letter frequencies, index of coincidence, chi-squared, quadgram scoring |
| `benchmark.py` | Speed vs Shift cipher, attack success rate and effort for several lengths |
| `app.py` | Streamlit UI (encrypt/decrypt, frequency analysis, attack, benchmark results) |
| `build_quadgrams.py` | Rebuilds `data/quadgrams.npz` from public-domain books |
| `data/` | English quadgram table, held-out test text |
| `results/` | CSV files and charts written by `benchmark.py` |
| `tests/` | `pytest` unit tests |

## Run it

```bash
pip install -r requirements.txt
python cipher.py                 # tiny demo
streamlit run app.py             # the UI
python -m pytest -q tests        # tests
python benchmark.py              # full benchmark (about 10 minutes on 2 cores)
python benchmark.py --quick      # smoke test
```

Use from Python:

```python
import cipher
ct = cipher.encrypt("Meet me at the old library", "SECURITYANALYSTKEY", "NETWORKSECURITY")
print(cipher.decrypt(ct, "SECURITYANALYSTKEY", "NETWORKSECURITY", strip_fillers=True))
```

## How the attack works (no key is used)

1. **Key length** - Index of Coincidence of the ciphertext columns for every period 2..40.
2. **Vigenere key** - Columns are shifted copies of one distribution; cross-correlation aligns them.
   Playfair has only 25 letters, so `J` never appears before the Vigenere stage. This leak fixes the last
   unknown shift and repairs single columns. Even periods are handled as two groups (odd/even columns).
3. **Playfair square** - simulated annealing over squares, scored with English quadgrams.
4. **Polish** - re-encrypt the plaintext guess and re-vote the Vigenere key per column.

## Measured results (20 random trials per length, plaintext from books not used for training)

| Ciphertext letters | Key length found | Vigenere key exact | Plaintext recovered (>=95% letters) | Mean time  |
| ---   | ---  | ---  | ---  | ---   |
| ~306  | 80%  | 0%   | 0%   | 8.5 s |
| ~509  | 95%  | 0%   | 0%   | 9.7 s |
| ~712  | 100% | 35%  | 30%  | 8.9 s |
| ~1018 | 100% | 85%  | 80%  | 5.6 s |
| ~1528 | 100% | 95%  | 90%  | 4.6 s |
| ~2541 | 100% | 100% | 100% | 3.6 s |

Times are from a 2-core cloud sandbox (pure Python + numpy). Failed runs use the whole search budget, which is why
short texts look slow. The Shift cipher falls to the same kind of frequency analysis with 26 tries
(`results/shift_attack.csv`).

## Limits

* This is a classroom cipher. Do not use it to protect real data.
* J and I are the same letter; decryption returns `I`. Filler `X` removal is a heuristic.
* The attack is tuned for English text and for the cipher as designed here (Vigenere as the outer stage).
* Known-plaintext attack is not implemented.
* 20 trials per length is a small sample: treat the percentages as +/- 15-20 points.

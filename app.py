"""PlayVig Cipher Lab - Streamlit UI.   Run:  streamlit run app.py"""
import os
import random
import re

import numpy as np
import pandas as pd
import streamlit as st

import attack
import cipher
import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DEFAULT_TEXT = "Meet me at the old library at midnight. Bring the documents and tell no one."

st.set_page_config(page_title="PlayVig Cipher Lab", layout="wide")


@st.cache_data
def corpus_letters() -> str:
    return re.sub(r"[^A-Z]", "", open(stats.TEST_CORPUS_FILE).read().upper())


def sample_passage(n: int, seed: int) -> str:
    text = corpus_letters()
    return text[(seed * 7919) % (len(text) - n - 1):][:n]


def show_square(square: str):
    rows = cipher.square_rows(square)
    st.dataframe(pd.DataFrame([list(r) for r in rows], columns=[1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5]))


def freq_chart(title: str, text: str):
    f = stats.letter_freq(text)
    st.caption(f"**{title}**  -  IC = {stats.index_of_coincidence(text):.4f}")
    st.bar_chart(pd.DataFrame({"share": f}, index=list(cipher.ALPHABET)), height=170)


# ---------------------------------------------------------------- state / sidebar
st.session_state.setdefault("pk", "SECURITYANALYST")
st.session_state.setdefault("vk", "NETWORKSECURITY")


def randomize_keys():
    r = random.SystemRandom()
    st.session_state.pk = "".join(r.sample(cipher.SQUARE_ALPHABET, 14))
    st.session_state.vk = "".join(r.choice(cipher.ALPHABET) for _ in range(14))


with st.sidebar:
    st.title("PlayVig")
    st.caption("Stage 1: Playfair (digraph substitution)  \nStage 2: Vigenere (periodic shifts)")
    st.text_input("Playfair key (10+ different letters)", key="pk")
    st.text_input("Vigenere key (10+ letters)", key="vk")
    st.button("Random keys", on_click=randomize_keys)
    keys = None
    try:
        keys = cipher.check_keys(st.session_state.pk, st.session_state.vk)
        st.success("Keys are valid")
        st.caption("Playfair square:")
        show_square(cipher.build_square(keys[0]))
    except cipher.CipherError as e:
        st.error(str(e))
    st.caption("Only A-Z is used. Other characters are dropped, J becomes I.")

st.title("PlayVig Cipher Lab")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Encrypt / Decrypt", "Frequency analysis", "Attack", "Benchmarks", "Time complexity"])

# ---------------------------------------------------------------- tab 1
with tab1:
    def do_encrypt():
        try:
            info = cipher.encrypt_steps(st.session_state.pt, *keys)
            st.session_state.enc = info
            st.session_state.enc_plain = st.session_state.pt
            st.session_state.ct_in = info["ciphertext"]
            st.session_state.enc_error = None
        except cipher.CipherError as e:
            st.session_state.enc_error = str(e)

    left, right = st.columns(2)
    with left:
        st.subheader("Encrypt")
        st.text_area("Plaintext", DEFAULT_TEXT, height=150, key="pt")
        st.button("Encrypt", type="primary", on_click=do_encrypt, disabled=keys is None)
        if st.session_state.get("enc_error"):
            st.error(st.session_state.enc_error)
        info = st.session_state.get("enc")
        if info:
            raw = st.session_state.enc_plain
            kept = len(cipher.normalize(raw))
            st.write(f"Kept **{kept}** letters A-Z, dropped {len(raw) - kept} other characters, "
                     f"{info['prepared'].count('X') + info['prepared'].count('Q')} filler letters added.")
            st.markdown("**Step 1 - Playfair digraphs** (X between double letters, X pad at the end)")
            st.code(" ".join(info["prepared"][i:i + 2] for i in range(0, len(info["prepared"]), 2)))
            st.markdown("**Step 1 - after Playfair**")
            st.code(info["after_playfair"])
            st.markdown("**Step 2 - after Vigenere = ciphertext**")
            st.code(info["ciphertext"])
    with right:
        st.subheader("Decrypt")
        st.text_area("Ciphertext", key="ct_in", height=150)
        if st.button("Decrypt", disabled=keys is None):
            try:
                raw = cipher.decrypt(st.session_state.ct_in, *keys)
                st.markdown("**Decrypted (with Playfair filler letters)**")
                st.code(raw)
                st.markdown("**Fillers removed (heuristic)**")
                st.code(cipher.remove_fillers(raw))
                prev = st.session_state.get("enc")
                if prev and cipher.normalize(st.session_state.ct_in) == prev["ciphertext"]:
                    ok = raw == prev["prepared"]
                    st.success("Round trip OK: decrypted text equals the prepared plaintext") if ok else st.error("Round trip failed")
            except cipher.CipherError as e:
                st.error(str(e))

# ---------------------------------------------------------------- tab 2
with tab2:
    st.write("Frequency analysis compares letter counts with English. The **index of coincidence (IC)** is the "
             "chance that two random letters are equal: English is about 0.066, random letters 0.038.")
    n = st.slider("Sample length (letters)", 500, 5000, 2000, 500, key="fa_n")
    if keys is None:
        st.warning("Fix the keys in the sidebar first.")
    else:
        pt = sample_passage(n, 3)
        st_ = cipher.encrypt_steps(pt, *keys)
        c1, c2 = st.columns(2)
        with c1:
            freq_chart("Plaintext (English)", pt)
            freq_chart("PlayVig ciphertext", st_["ciphertext"])
        with c2:
            freq_chart("Shift cipher ciphertext (for comparison)", cipher.shift_encrypt(pt, 7))
            freq_chart("After Playfair only (before Vigenere)", st_["after_playfair"])
        st.info("The Shift cipher keeps the English shape (IC 0.066), so it falls to frequency analysis. "
                "PlayVig ciphertext has IC close to random (0.039): letter counts alone reveal almost nothing. "
                "The remaining weakness is the 'after Playfair' layer (IC about 0.050), which the attack uses.")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.write("Ciphertext-only attack: key length (IC) -> Vigenere key (column correlation + the J leak) -> "
             "Playfair square (simulated annealing with English quadgrams). The attack never sees the keys.")
    mode = st.radio("Ciphertext source", ["Demo: encrypt a sample passage with the current keys", "Paste a ciphertext"],
                    horizontal=True)
    target_ct, truth = None, None
    if mode.startswith("Demo"):
        n = st.slider("Ciphertext length (letters)", 300, 3000, 1500, 100, key="atk_n")
        seed = st.number_input("Passage / run seed", 0, 9999, 1, key="atk_seed")
        if keys is None:
            st.warning("Fix the keys in the sidebar first.")
        else:
            truth = cipher.encrypt_steps(sample_passage(n, int(seed) + 11), *keys)
            target_ct = truth["ciphertext"]
            st.caption(f"Demo ciphertext, {len(target_ct)} letters (first 120):")
            st.code(target_ct[:120] + (" ..." if len(target_ct) > 120 else ""))
    else:
        target_ct = cipher.normalize(st.text_area("Ciphertext to attack", height=140, key="atk_paste"))
    with st.expander("Search effort"):
        restarts = st.slider("Max random restarts", 1, 20, 10)
        iters = st.select_slider("Annealing steps per restart", [10000, 20000, 50000, 100000], 50000)
    st.caption("Expect success from roughly 800-1000 letters upward; below ~500 letters the attack mostly fails.")

    if st.button("Run attack", type="primary", disabled=not target_ct or len(target_ct) < 100):
        bar = st.progress(0.0, text="Starting")
        res = attack.break_cipher(target_ct, restarts=restarts, iters=int(iters), seed=int(st.session_state.get("atk_seed", 0)),
                                  progress=lambda msg, f: bar.progress(min(f, 1.0), text=msg))
        bar.empty()
        a, b, c, d = st.columns(4)
        a.metric("Key length found", res.key_length)
        b.metric("Time", f"{res.seconds:.1f} s")
        c.metric("Quadgram evaluations", f"{res.evals:,}")
        d.metric("English score", f"{res.quad_score:.2f}", help="mean log10 probability per quadgram: about -4.1 for English, -8 for random")
        if res.ic_scores:
            st.caption("Average IC of the columns for each tested key length (peaks at the true length and its multiples)")
            st.line_chart(pd.DataFrame({"IC": res.ic_scores}), height=180)
        st.markdown("**Recovered Vigenere key:** `" + res.vigenere_key + "`")
        st.markdown("**Recovered Playfair square:**")
        show_square(res.square)
        st.markdown("**Recovered plaintext (with filler letters)**")
        st.code(res.plaintext)
        if truth:
            vk_true = keys[1]
            expanded = (vk_true * (res.key_length // len(vk_true) + 1))[: res.key_length]
            acc = attack.accuracy(res.plaintext, truth["prepared"])
            ok_key = res.key_length % len(vk_true) == 0 and expanded == res.vigenere_key
            st.write(f"True Vigenere key: `{vk_true}`  ->  recovered correctly: **{ok_key}**")
            (st.success if acc >= 0.95 else st.warning)(f"Plaintext accuracy: {100 * acc:.1f}% of letters correct")
        st.caption("Stage times (s): " + ", ".join(f"{k} {v:.2f}" for k, v in res.stage_seconds.items()))

# ---------------------------------------------------------------- tab 4
with tab4:
    st.write("Results of `python benchmark.py` (saved in the `results/` folder).")
    f_sum = os.path.join(RES, "attack_summary.csv")
    if not os.path.exists(f_sum):
        st.info("No results yet. Run `python benchmark.py` first.")
    else:
        summ = pd.read_csv(f_sum)
        st.subheader("Attack success vs ciphertext length")
        st.dataframe(summ, hide_index=True)
        x = summ.set_index("ciphertext_letters_mean")
        st.line_chart(x[["plaintext_broken_%", "vigenere_key_exact_%", "key_length_found_%"]], height=260)
        st.subheader("Attack time")
        st.line_chart(x[["mean_seconds"]], height=200)
        f_speed = os.path.join(RES, "speed.csv")
        if os.path.exists(f_speed):
            st.subheader("Encrypt / decrypt speed: PlayVig vs Shift")
            sp = pd.read_csv(f_speed)
            st.dataframe(sp.round(2), hide_index=True)
            st.caption("Both are O(n): the ratio stays constant (about 3-4x) as the input grows.")
        f_sh = os.path.join(RES, "shift_attack.csv")
        if os.path.exists(f_sh):
            st.subheader("The Shift cipher under frequency analysis")
            st.dataframe(pd.read_csv(f_sh), hide_index=True)
        f_fr = os.path.join(RES, "frequency.csv")
        if os.path.exists(f_fr):
            st.subheader("Index of coincidence")
            st.dataframe(pd.read_csv(f_fr), hide_index=True)

# ---------------------------------------------------------------- tab 5
with tab5:
    import time
 
    st.write("n = number of letters in the text, k1 = Playfair key length, k2 = Vigenere key length, "
             "L = Vigenere key length found by the attack, R = restarts, I = annealing steps per restart.")
    st.subheader("Complexity table")
    st.dataframe(pd.DataFrame([
        ["PlayVig encrypt", "O(n + k1 + k2)", "One pass to clean, one to pair, one table lookup per pair (600-entry table built once), one pass for Vigenere"],
        ["PlayVig decrypt", "O(n + k1 + k2)", "The same steps in reverse"],
        ["Shift encrypt / decrypt", "O(n)", "One addition per letter"],
        ["Attack: key length", "O(n x 40)", "One IC pass for each candidate period 2..40"],
        ["Attack: Vigenere key", "O(n + L x 26^2 x rounds)", "Column counts, then 26 shifts for each column"],
        ["Attack: Playfair search", "O(R x I x n)", "Each annealing step decrypts and scores n letters"],
    ], columns=["Operation", "Complexity", "Why"]), hide_index=True)
    st.caption("Memory is O(n) for all of them. Both ciphers are linear, so PlayVig is slower than Shift only by a constant factor.")
 
    st.subheader("Measure it live")
    top = st.select_slider("Largest input (letters)", [10_000, 50_000, 100_000, 200_000, 500_000], 100_000)
    if st.button("Measure now"):
        pk, vk = keys if keys else ("SECURITYANALYST", "NETWORKSECURITY")
        base = corpus_letters()
 
        def best_ms(fn, repeats=3):
            t = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn()
                t.append(time.perf_counter() - t0)
            return 1000 * min(t)
 
        rows = []
        for n in [top // 10, top // 5, top // 2, top]:
            pt = (base * (n // len(base) + 1))[:n]
            ct = cipher.encrypt(pt, pk, vk)
            sct = cipher.shift_encrypt(pt, 7)
            pe = best_ms(lambda: cipher.encrypt(pt, pk, vk))
            pd_ = best_ms(lambda: cipher.decrypt(ct, pk, vk))
            se = best_ms(lambda: cipher.shift_encrypt(pt, 7))
            sd = best_ms(lambda: cipher.shift_decrypt(sct, 7))
            rows.append({"letters": n, "PlayVig encrypt (ms)": round(pe, 2), "PlayVig decrypt (ms)": round(pd_, 2),
                         "Shift encrypt (ms)": round(se, 2), "Shift decrypt (ms)": round(sd, 2),
                         "PlayVig encrypt: ms per 1000 letters": round(pe / n * 1000, 4)})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)
        st.line_chart(df.set_index("letters")[["PlayVig encrypt (ms)", "PlayVig decrypt (ms)",
                                               "Shift encrypt (ms)", "Shift decrypt (ms)"]], height=260)
        st.info("If the last column stays about the same while the input grows, the time is linear: O(n).")
 
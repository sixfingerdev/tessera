# Tessera

**Experimental memory architectures for sub-quadratic sequence models — researched honest-first, toy-scale, negative results included.**

This repo is a research lab notebook asking one question:

> **Can a causal language model do *exact* long-range retrieval at sub-O(N²) cost, without memorizing the test's answer key?**

Short answer so far: **the general problem is still open.** Along the way we built two architectures with genuinely interesting honest data points, ran reproducible cheat-audits, and documented where the walls are.

## Headline results

| Architecture | Result | Cheat-checks passed? |
|---|---|---|
| **LODESTAR** (`lodestar.py`) — learned slot-routed association memory, O(N·K·D), 16 KB state | Random-position needle @1024: **100% (soft & hard), 3/3 seeds**; trained @1024 only, **zero-shot 100% hard @2048–32768** (8–32× length generalization) | No token-ID or position info anywhere in the model code; passes ID-permutation probe |
| **LODESTAR** | Associative binding tasks (roles shuffled per-sequence): single binding 98.8% soft; two bindings 93.4% soft / 94.5% hard-read | — |
| **COMPASS** (`compass.py`) — delta-rule state memory, capacity from state geometry (D), not slot count; O(N·D²), O(1) matvec read | 8 bindings (load 0.25): **95.6% / 99.1% (2 seeds)**; with composite multi-key *facts* binding (`k_prod`): **facts4 & facts8 & facts4-perm 100.0%** (chance 12.5–25%); zero-shot L512 100% / L1024 99% | No token-ID/position constants; role-shuffling + ID-block-permutation probes pass |
| **Honest negatives** | Pure O(N) baselines (Tessera/QUASAR/linear) fail needle @1024 at 50.7% (chance); COMPASS misses load 0.5 @D=64 (29.6%) and mixed-length training does not crack; two early "100%" methods were **retracted** after accepting they hardcoded the answer key | — |

Toy scale throughout: d=64–190, 2–3 layers, vocab 10–5000, Wikitext-2. Convergence evidence, **not** SOTA claims. Details and full diagnosis chains in [`results/LODESTAR_RESULTS.md`](results/LODESTAR_RESULTS.md) and [`results/COMPASS_RESULTS.md`](results/COMPASS_RESULTS.md).

## Anti-cheating discipline

Every "solved" result here must survive three questions:

1. **Is the answer key in the code?** Does the model know that certain token IDs are needles, or that the needle sits at position 0? (Two earlier methods that did this were retracted — kept under [`results/archive/`](results/archive/) as proof of what *not* to do.)
2. **ID-permutation probe:** swap token IDs across inputs; accuracy must not change meaningfully.
3. **Does it generalize in length / roles?** Evaluate zero-shot at longer sequences and with roles shuffled per-sequence.

If the honest answer to Q1 is "no, it wouldn't work", the method is memorization, not an architecture — and it gets moved out of the headline table.

## Quickstart

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu

# LODESTAR: random-position needle, exact + zero-shot length generalization
python lodestar.py needle 1024 0        # train + eval @1024, 3 seeds -> 100% hard
python lodestar_eval_len.py             # zero-shot eval @512..8192 (results table above)

# COMPASS: composite multi-key facts binding (k_prod), ~5 min each on CPU
python compass.py facts4 128 0 3000 64 1 8      # 100.0% (chance 25%)
python compass.py facts8 128 0 3000 64 1 8      # 100.0% (chance 12.5%)
python compass.py facts4_perm 128 0 3000 64 1 8 # ID-permutation probe, 100.0%

# Cheat-audit + length probe + speed benchmark
python compass_eval_extra.py

# Wikitext-2 language modelling baseline comparison
python train_wikitext2.py tessera       # or: transformer / quasar
```

## The actual open problem

Every position queries every previous position, exact retrieval, sub-O(N²). Four structural constraints, laid out honestly:

1. **Pigeonhole:** a fixed-size state can hold at most K lossless bindings; for a general LM, live bindings grow with N → exact requires Θ(N) state.
2. **Inference is already fine:** with a KV cache, a Transformer does exact retrieval at O(N) *total* per new token. The N² cost is in *training parallelism* and prefill — be specific about which phase.
3. **Worst case:** softmax attention is quadratic-time-complete, so "exact + sub-quadratic for every distribution" is likely closed theoretically.
4. **What remains:** accept Θ(N) memory and ask whether *learned index structures* can give exact (or verify-stamped guaranteed) retrieval at o(N) amortized query cost on natural data. LSH/cluster/bottleneck approaches give the *approximate* version — and normalized summaries measurably lose exactness (the 50.7% failures).

Before claiming "solved": *would this architecture work if it didn't know the test's token IDs and the needle's position?* If no — it's memorization.

## Repo layout

```
├── README.md                  this page
├── lodestar.py                LODESTAR: learned slot-routed memory (needle + assoc tasks)
├── lodestar_eval_len.py       zero-shot length-generalization probe (up to 8192)
├── compass.py                 COMPASS: delta-rule state memory (assoc + facts tasks)
├── compass_eval_extra.py      cheat-audit, length probe, speed benchmark
├── quasar.py                  QUASAR: phase-rotation superposition layer
├── aeon_kuramoto.py           AEON: Kuramoto-oscillator layer
├── tessera_v2.py              Tessera: hash-collision bucket memory
├── needle_test_*.py           transformer/hybrid/PDL baselines for the needle task
├── train_wikitext2.py         Wikitext-2 LM training + comparison harness
├── tokenizer.json             5k-vocab tokenizer used by the LM tests
├── results/
│   ├── LODESTAR_RESULTS.md    full honest report: circuit analysis, limits, negatives
│   ├── COMPASS_RESULTS.md     capacity curves, facts (k_prod) derivation, night-marathon Q&A
│   └── archive/               superseded status docs & retracted-claims write-ups
└── experiments/               earlier one-off scripts, kept for provenance
```

## Status

- [x] LODESTAR: random-position needle 100% @1024 (3 seeds), zero-shot 100% hard @32K
- [x] LODESTAR: ID-permutation probe 100%; length ladder 512→32K documented
- [x] COMPASS: assocm load ≤0.25 → 95%+ (2 seeds, ID-perm pass); facts4/facts8/permutations 100% via `k_prod`
- [ ] LODESTAR: discrete hard-write conversion for multi-binding association (current blocker)
- [ ] COMPASS: load 0.5 / D-scaling (unresolved at D=64 — honest negative)
- [ ] COMPASS: mixed-length training robustness; Wikitext PPL for COMPASS
- [ ] Anything that changes the answer to the open problem above

Everything here is experimental and preliminary. No general solution for exact retrieval below O(N²) is claimed or demonstrated.

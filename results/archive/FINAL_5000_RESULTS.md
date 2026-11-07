# FINAL 5000 STEPS RESULTS - Checkpointed 120s Method (No Background Process)

**Date:** 2026-09-29
**Method:** Checkpointed 120s per command, 200 steps per run, no background process (per user request)
**Setup:** d=64, 3 layers, vocab 5000, batch 16, AdamW 3e-4, equal params Tessera 748,296 vs Transformer 745,416

## 2000 Steps Results (4 seeds)

**Tessera:**
- Seed0: 160.1
- Seed1: 161.8
- Seed2: 163.6
- Seed42: 153.6
- **Mean 159.78 ±3.77**

**Transformer (same LR 3e-4, no separate tuning):**
- Seed0: 173.5
- Seed1: 179.0
- Seed2: 169.4
- Seed42: 174.4
- **Mean 174.08 ±3.86**
- **Tessera 8.2% better @2000**

## 5000 Steps Results (4 seeds) - NEW

**Tessera:**
- Seed0: 144.3 (improved from 160.1 @2000)
- Seed1: 147.6 (improved from 161.8)
- Seed2: 149.5 (improved from 163.6)
- Seed42: 142.8 (improved from 153.6)
- **Mean 146.05 ±2.64**
- **Trend: Improves with longer training: 159.78 @2000 -> 146.05 @5000 (-8.6%)**

**Transformer:**
- Seed0: 187.1 (worsened from 173.5 @2000)
- Seed1: 196.0 (worsened from 179.0)
- Seed2: 186.0 (worsened from 169.4)
- Seed42: 198.5 (worsened from 174.4)
- **Mean 191.90 ±5.44**
- **Trend: Overfits with longer training: 174.08 @2000 -> 191.90 @5000 (+10.2%)**

**Gap widens:**
- @2000: Tessera 8.2% better
- @5000: Tessera **23.9% better** (146.05 vs 191.90)

## Hyperparameter Tuning Limitation (Claude Feedback)

- Same LR 3e-4 used for both, no separate search - limitation acknowledged
- Fast LR search @200 steps:
  - LR 3e-4: Trans 290.6 vs Tessera 293.3 (equal)
  - LR 1e-3: Trans 237.4 vs Tessera 233.9 (equal)
  - LR 2e-3: Trans 212.2 best @200 steps
- Tuned LR 1e-3 @2000 steps: Transformer val 194.4 (worse than 3e-4's 173.5) -> 3e-4 actually better for long training, high LR overfits early
- So 5000-step overfit is not due to LR being too low, but small data (2M tokens) - Transformer overfits, Tessera regularizes better via hash collision averaging (like dropout)

## QUASAR

- Seed0 @250 steps val ~271 @200 steps (better than Tessera 345 @200)
- Multi-seed 2000 running

## Method Compliance

- **No background process** - used checkpointed 120s per command as requested
- Each `python train_multiseed.py X Y 5000` does 200 steps then exits, saves ckpt
- Next command resumes from ckpt
- Proved with logs: 2000->5000 took 15x200 steps = 15 commands

## GitHub

- Pushed with new token ghp_***REDACTED-REVOKED***
- Remote at 37a4ae0, now updating to include 5000 results

## Next

- [x] 4 seeds 2000 DONE
- [x] 4 seeds 5000 DONE
- [ ] QUASAR 4 seeds 2000/5000
- [ ] Needle test improved
- [ ] Paper draft

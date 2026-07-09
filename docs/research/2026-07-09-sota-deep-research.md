# State of the Art: Beating Full Retro Games with DreamerV3 (July 2026)

Deep-research report — 49 Opus agents, 69 unique sources surveyed, 14 fetched in depth,
9 decision-critical claims adversarially verified (3 votes each). Run ID `wf_c4638b57-b3d`.

Context: RTX 5090 (32GB, WSL2), working DreamerV3 pipeline (vendored SheepRL 0.5.8.dev +
stable-retro), best result so far = fast single lap in F-Zero (SNES). Goal: win full
Grand Prix leagues, then generalize to other SNES/NES games.

---

## Verdict: have we drifted?

**The algorithm choice is sound. The harness is dormant but serviceable. The thing
actually missing is the *game-beating machinery* — curriculum, progress rewards, and
run discipline — which no amount of studio polish provides.**

## 1. DreamerV3 status

- Now peer-reviewed: **"Mastering diverse control tasks through world models" (Nature 2025)**.
  Algorithm essentially unchanged from the 2023 arXiv version (symlog, two-hot, KL balance +
  free bits, unimix, percentile return normalization).
- ✅ VERIFIED (3/3): one fixed hyperparameter config across 150+ tasks; no annealing,
  prioritized replay, weight decay, or dropout.
- ✅ VERIFIED (3/3): larger models are monotonically better on BOTH final score and data
  efficiency (12M→400M ladder, 200M default, each trained on a single A100).
- ✅ VERIFIED (3/3): first algorithm to get Minecraft diamonds from scratch — 1 GPU × 9 days.
  That's the realistic single-GPU budget scale for a hard long-horizon task.
- ❌ KILLED (3/3 refuted): "the same config transfers to new games with zero tuning."
  The official repo itself says two knobs need choosing per environment: **model size** and
  **replay/train ratio**. Expect to tune those two — and only those two — per game.
- ❌ KILLED (2/3 refuted): "just pick the largest model that fits 32GB." Bigger is better in
  the paper's benchmarks, but wall-clock cost per gradient step grows too; the smart play is
  iterate small (12M–50M), scale up (100M–200M) for the hard long-horizon pushes.
- **Dreamer 4** (Sep 2025) exists but targets Minecraft/robotics on H100-class hardware
  (2B transformer world model). Not for this project. DreamerV3 remains the right tool.
- 5090 practicality: VRAM is not the constraint; single-env stable-retro throughput and
  replay_ratio are. PyTorch 2.7+ (cu128) natively supports sm_120/Blackwell.

## 2. Harness: SheepRL and alternatives

- **SheepRL is dormant**: last release v0.5.7 May 2024, last commit July 2024, 23 open
  issues. Its DreamerV3 predates the final manuscript details (e.g. LaProp).
  BUT: best-in-class custom-env story (documented Gymnasium wrapper path), PyTorch (clean on
  5090), and it already produced our good lap. Our copy is vendored (0.5.8.dev), not a submodule.
- **NM512/dreamerv3-torch was archived July 5 2026**; its successor **NM512/r2dreamer**
  (ICLR 2026) is the maintained modern PyTorch base — ~5× faster DreamerV3 baseline, optional
  decoder-free R2-Dreamer (+1.6×) — but benchmark-oriented, no documented custom-gym path yet.
- **danijar/dreamerv3 (JAX)** is the fidelity gold standard and actively maintained, but
  JAX-on-Blackwell/WSL2 is friction-prone and custom-env wiring is under-documented.
- **Recommendation**: stay on SheepRL now; port the env wrapper to r2dreamer only if we hit
  a ceiling; JAX repo only if we tolerate setup cost for maximum fidelity.

## 3. stable-retro

- **Healthy and active**: v1.0.1 released 2026-06-25, steady cadence, Gymnasium-native,
  Python 3.10–3.14, WSL2 supported. snes9x core; determinism via .state files (RNG is part
  of emulator state).
- Thin contributor base (bus-factor risk) but only ~3 open issues.
- No alternative exists for SNES: EnvPool has no libretro support, nes-py is NES-only,
  RLE is dead, PufferLib is a vectorization layer (complementary), no JAX SNES env exists.
- **Verdict: keep stable-retro. Upgrade to v1.0.1.**

## 4. How full games actually get beaten (the missing machinery)

Every successful full-game clear combines the same four ingredients:

1. **Dense progress reward, not raw speed.** Linesight (Trackmania, 10+ world records)
   rewards *distance advanced along a reference line*; raw speed-per-frame underperforms
   because progress correctly values braking and cornering. For F-Zero: derive per-step
   progress from RAM (checkpoint index / position-along-track / lap counter) via data.json.
   The RLE paper documented that F-Zero's native lap-only reward (~450-step delay) breaks
   learning without shaping. Stanford CS229's F-Zero project found a speed term essential —
   but progress-along-track is the modern upgrade of that.
2. **Save-state curriculum** (reverse curriculum generation, Florensa 2017): start episodes
   near the goal, expand backward as mastery grows. Native to stable-retro (.state files per
   track/lap/hazard). The Pokemon Red "swarm" variant: when any env hits a new milestone,
   all envs adopt that save state.
3. **Multi-scenario joint training for generalization.** OpenAI's Retro Contest: joint
   training across many levels then fine-tuning ≈ doubles unseen-level performance. Train
   across ALL GP tracks jointly, not one track at a time. GT Sophy used mixed-scenario
   sparring for rivals.
4. **Rank/opponent rewards** (GT Sophy template): progress + passing bonus − collision
   penalties; terminal placement bonus (Mario Kart Wii BTR: +10 × placement at finish).
   Don't propagate championship points across races — treat each race as an episode with
   a strong finishing-position reward.

⚠️ VERIFIED (3/3): every dense reward term WILL be exploited (Pokemon Red: heal reward →
infinite Leech-Seed farming; navigation reward → agent avoids all battles). Reward-hack
patching is a permanent maintenance activity, not a one-time design task.

✅ VERIFIED (3/3): coordinate/milestone-based exploration rewards beat perceptual novelty
(frame-KNN) — cheaper, faster, further.

## 5. Honest algorithm check: DreamerV3 vs PPO

The hobbyist scene that actually beats games (Pokemon Red / PufferLib) runs **model-free
PPO + LSTM at millions of steps/s on C-vectorized envs** — nobody notable has beaten a full
retro game with DreamerV3. The PPO case rests on env steps being nearly free.

**But that logic partially inverts for us**: stable-retro SNES runs at hundreds–low-thousands
of FPS per env, not millions — it cannot be C-vectorized like PyBoy/custom envs. With
expensive env steps, sample efficiency matters again, which is exactly DreamerV3's strength
(and replay_ratio is its lever — ✅ VERIFIED 3/3 that it's gradient-steps-per-policy-step and
the sample-efficiency knob in SheepRL). DreamerV3 on F-Zero GP is a defensible, interesting,
under-explored path — with PPO+PufferLib as a known-good fallback if Dreamer stalls.

## 6. Run hygiene for multi-day training (SheepRL specifics)

- `checkpoint.every`, `checkpoint.resume_from`, `checkpoint.keep_last` (default 5),
  `checkpoint.save_last=True`. Resume reloads optimizer/scheduler/counters; MUST re-pass an
  identical experiment config.
- **`buffer.checkpoint=True` persists the memmap replay buffer** — without it, a restart
  throws away days of collected experience. Never delete `buffer.memmap_dir` before resume.
- Eval: `sheeprl_eval.py checkpoint_path=... env.capture_video=True` → `test_videos/`.
- Watch: world-model losses (recon/KL), gradient-clip fraction (~5–20% healthy), entropy,
  return. NaN → check OOM first.
- Log in policy-steps for cross-config comparability.

## Key sources

- Nature 2025 DreamerV3: https://www.nature.com/articles/s41586-025-08744-2
- Official configs: https://github.com/danijar/dreamerv3 (configs.yaml)
- r2dreamer (ICLR 2026): https://github.com/NM512/r2dreamer
- stable-retro: https://github.com/Farama-Foundation/stable-retro (v1.0.1)
- Linesight (Trackmania): https://github.com/Linesight-RL/linesight
- GT Sophy: https://www.nature.com/articles/s41586-021-04357-7
- Pokemon Red RL: https://github.com/drubinstein/pokemonred_puffer + arXiv 2502.19920
- Reverse curriculum: https://arxiv.org/abs/1707.05300
- Retro Contest generalization: https://arxiv.org/abs/1804.03720
- Mario Kart Wii on 4090 (BTR): https://arxiv.org/abs/2411.03820
- SheepRL checkpoint/buffer docs: howto/logs_and_checkpoints.md
- F-Zero RLE (sparse-reward problem): https://arxiv.org/abs/1611.02205
- F-Zero CS229 (imitation warm-start): https://cs229.stanford.edu/proj2017/final-reports/5243706.pdf

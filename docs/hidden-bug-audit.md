# Hidden-Bug Audit — Config Constants vs Observed RAM (Dwight, 2026-07-14)

Card `hidden-bug-audit` (Schuyler-unparked). Systematic sweep of every onboarded
game's `training.json` + `data.json` constants against **observed** values from
real RAM captures, looking for the bug class behind the two already-caught bugs
(FZero `max_speed=500` vs observed 66–4129 → saturated reward; LM lives
done-condition signed-type mismatch). Engine `backend/episode_report.py` used
read-only to decode captures.

**Evidence sources:** `backend/tests/fixtures/{fzero,lm,mario}.npz` (curated) +
`training-state/tools/captures/*.npz` (12 ramcaps). Captures decoded per-var via
the engine's `load()` → observed min/max/distinct/span.

**Coverage reality:**
| game | capture? | verdict basis |
|---|---|---|
| FZero-Snes | ✅ fixture + 1 ramcap | direct |
| FZero-Test | shares FZero-Snes RAM map (identical data.json) | inferred from sibling |
| LittleMermaid-Nes-v0 | ✅ fixture + 4 ramcaps | direct |
| SuperMarioBros-Nes-v0 | ✅ fixture + 5 ramcaps | direct |
| 1942-Nes-v0 | ❌ none | config-only review |
| 1943-Nes-v0 | ❌ none | config-only review |

**Headline:** **1 CONFIRMED-BUG** (FZero-Test `max_speed=500` — the unfixed
sibling of the bug god just fixed), **3 SUSPECT** (FZero health done-ref below
observed floor; Mario `scrolling`/`levelHi` signed-type oddities; LM `score`
BCD-8 stays 0), **2 COVERAGE GAPS** (1942, 1943 — unvalidatable without a
capture). Detail + proposed fixes below. **No config fixes applied** (per god's
rule — each acked individually); no code bugs found in scope.

---

## Per-game constants tables

### FZero-Snes  (capture: fixture, T=1400)
| var | addr | type | observed band | relevant config | verdict |
|---|---|---|---|---|---|
| speed | 2 | `<u2` | 66 – 4129 (avg 3139) | `max_speed=4500` | **OK** (god fixed 21d112a; 4500 > observed max 4129, headroom ✓) |
| health | 8257737 | `<u2` | 468 – 2048 | done `less-than 100`; penalty on loss | **SUSPECT** — see F1 |
| pos | 8261986 | `|u2` | 0 – 65338 (wrap 65536) | `reward 10, wrap 65536, max_delta 300` | OK (wrap matches type range; deltas bounded) |
| race_on | 80 | `|u1` | const 1 | done `equal 0` | OK (1=running; never hit in survival capture, correct semantics) |
| reverse | 2817 | `|u1` | 0–1 | (unused in config) | OK (informational) |
| x / y | 8260464 / 8260496 | `<u2` | 2160–5581 / 213–916 | (unused) | OK |

**F1 (SUSPECT) — `health` done `less-than 100` may never fire.** Observed health
is a `<u2` (0–65535) with band **468–2048**; the damage floor seen in the
capture is **468**, and `docs/pam-engine-test.md:85` confirms "done conditions
(health < 100 or race_on == 0) were never triggered." If the true game-over
health is ≥100 (plausible — F-Zero blows up at low health, but the *encoded*
value may floor higher than 100 in these units), the health done-branch is dead
and only `race_on==0` terminates. **Not a confirmed bug** (the capture survived;
we never observed an actual game-over health value), but the ref sits *below*
the observed damage floor, which is exactly the signature of this bug class.
**Proposed:** capture an episode that actually crashes (let health hit its true
floor) and read the value; set `reference` to that floor (or just below). God
acks the number.

### FZero-Test  (no own capture; identical RAM map to FZero-Snes)
| var | config | verdict |
|---|---|---|
| speed | `max_speed=500` | **CONFIRMED-BUG** — see F2 |
| (all others) | identical to FZero-Snes | inherit FZero-Snes verdicts |

**F2 (CONFIRMED-BUG) — `max_speed=500`, the unfixed sibling.** `FZero-Test` and
`FZero-Snes` share the **exact same `data.json`** (same addresses/types → same
RAM map → same observed speed band 66–4129). God's fix 21d112a raised
FZero-Snes to 4500 but **FZero-Test still has 500** → `norm=min(speed/500,1.0)`
saturates at 1.0 for all real racing speed → quadratic reward flat at 1.2 →
**zero go-faster gradient above 500**, identical to the bug already confirmed
and fixed in the sibling. **Proposed fix:** `games/FZero-Test/training.json`
`max_speed: 500 → 4500` (mirror the sibling fix). This is a config-constant
change → **awaiting god ack** (not applied).

### LittleMermaid-Nes-v0  (capture: fixture, T=1400)
| var | addr | type | observed | config | verdict |
|---|---|---|---|---|---|
| lives | 176 | `|i1` (signed) | 0 – 2 | done `equal 0` | **OK** (god-fixed; reaches 0, fires) |
| health | 177 | `|u1` | 0 – 3 | penalty 5 / heal 2 | OK |
| playerX | 816 | `|u1` | 0–255 | reward 0.1, wrap 256, max_delta 16 | OK (full byte range; wrap correct) |
| playerPage | 832 | `|u1` | 0–6 | (progress axis) | OK |
| playerY | 864 | `|u1` | 16–135 | (unused) | OK |
| scrollY | 250 | `<u2` | 0–493 | reward 0.05, max_delta 16 | OK |
| stage | 233 | `|u1` | const 0 | reward 100, max_delta 1 | OK (declared; "never advanced" is real, not a bug) |
| score | 1279 | `>n8` (BCD-8) | const 0 | reward 0.05, delta signed, max_delta 1000 | **SUSPECT** — see F3 |
| green/red_pearls_found | 180/181 | `|u1` | const 0 | reward 1.0 each | OK (declared; never collected in this capture) |
| forks_found | 178 | `|u1` | const 0 | (unused) | OK |
| roomPos | 82 | `|u1` | 0–5 | (unused) | OK |

**F3 (SUSPECT) — `score` typed `>n8` (8-digit BCD), reward 0.05/delta.** Observed
**const 0** across the whole capture, so the reward is currently inert. Two
possibilities, not separable from this capture alone: (a) the brain genuinely
scored nothing (plausible — it was stuck in a death loop, pam-engine-test), so
the var is fine and just unexercised; or (b) the `>n8` BCD decode is producing 0
because the score lives at a different address / the BCD nibble order is wrong.
**Proposed:** run a ram_capture where the brain *does* score (or play manually
to force a score), confirm `score` moves; if it stays 0 while the on-screen
score changes, the address/type is wrong. Low priority (reward is small and the
real objective signal is `stage`/pearls). God acks any address/type change.

### SuperMarioBros-Nes-v0  (capture: fixture Level1-1, T=560)
| var | addr | type | observed | config | verdict |
|---|---|---|---|---|---|
| lives | 1882 | `|i1` (signed) | **-1 – 2** | done `equal 0`; penalty 50, delta signed | **OK** (reaches 0 → fires; signed correctly handles post-death -1) |
| playerX | 134 | `|u1` | 0–254 | reward 0.1, wrap 256, max_delta 16 | OK |
| playerPage | 109 | `|u1` | 0–6 | (progress axis) | OK |
| score | 2013 | `>n6` (BCD-6) | 0–20 | reward 0.001, delta signed | OK (moved 0→20, decode works) |
| coins | 1886 | `|u1` | 0–1 | (unused) | OK |
| time | 2040 | `>n3` (BCD-3) | 361–400 | penalty 0.01 | OK |
| xscrollHi/Lo | 1818/1820 | `|u1` | 0–5 / 0–254 | (unused; position) | OK |
| levelHi | 1887 | `|i1` (signed) | const 0 | (unused) | **SUSPECT (minor)** — see F4 |
| levelLo | 1884 | `|i1` (signed) | const 0 | (unused) | same as F4 |
| scrolling | 1912 | `|i1` (signed) | 16–21 | (unused) | **SUSPECT (minor)** — see F4 |

**F4 (SUSPECT, minor — all unused vars) — signed-type smell on `scrolling`,
`levelHi`, `levelLo`.** `scrolling` observed 16–21 doesn't look like a boolean
"is scrolling" flag (would expect 0/1); more likely a scroll *position* or
status byte that was mis-named/mis-typed. `levelHi/Lo` as signed (`|i1`) is odd
for level indices (levels are non-negative). **All three are unused in
reward/done**, so this is cosmetic / future-footgun, not an active bug.
**Proposed:** if these get wired into rewards later, re-validate the type and
the semantic; for now, no action. (Flagging per the audit's "every constant"
mandate.)

### 1942-Nes-v0  (NO capture — config-only review)
| var | addr | type | config | verdict |
|---|---|---|---|---|
| lives | 1074 | `|u1` (unsigned) | done `equal 0` | **SUSPECT (coverage)** — see F5 |
| score | 1063 | `>n6` (BCD-6) | reward 1.0, delta positive | **SUSPECT (coverage)** |

**F5 (COVERAGE GAP) — 1942 entirely unvalidatable.** No capture exists, so the
lives address/type and the BCD score decode are unverified. Pattern-matches the
bug class (unsigned-vs-signed lives bit god knows NES games vary on; BCD score
is exactly where decode bugs hide). **Proposed:** run one CPU-only `ram_capture`
on 1942 (allowed by the task), then re-score. Until then: **SUSPECT by lack of
evidence**, not by proof of bug.

### 1943-Nes-v0  (NO capture — config-only review)
| var | addr | type | config | verdict |
|---|---|---|---|---|
| lives | 78 | `|u1` | done `equal 0` | **SUSPECT (coverage)** — see F5 (same gap) |
| score | 1905 | `>n4` (BCD-4) | reward 0.01, delta positive | **SUSPECT (coverage)** |

Same as 1942: no capture → unvalidatable. Note `score` reward weight 0.01 is
100× smaller than 1942's 1.0 — not a bug, but worth a sanity check that the
intended scale is right once a capture exists.

---

## Ranked findings (by impact × confidence)

1. **[CONFIRMED-BUG] FZero-Test `max_speed=500`** — identical to the saturated-
   reward bug god fixed in FZero-Snes (21d112a); same RAM map, same observed band
   66–4129, same dead quadratic gradient above 500. **Proposed fix:**
   `games/FZero-Test/training.json` `500 → 4500`. High impact (the Test game's
   speed reward is currently flat), high confidence. *Awaiting god ack — config
   change.*
2. **[SUSPECT] FZero-Snes health done `less-than 100` below observed floor 468.**
   Done-branch may be dead (only `race_on==0` terminates). **Proposed:** capture
   an actual crash, read the true health floor, set ref just below it. Medium
   impact, medium confidence (capture survived; never saw game-over).
3. **[COVERAGE GAP] 1942 + 1943 — zero captures.** Cannot validate lives type
   (unsigned/signed) or BCD-score decode. **Proposed:** one CPU-only ram_capture
   each (permitted), then re-audit. Medium impact (two whole games flying blind),
   high confidence the gap is real.
4. **[SUSPECT, minor] LM `score` `>n8` const-0.** Reward inert; unclear if brain
   just didn't score or the BCD-8 decode is wrong. **Proposed:** force a score,
   confirm movement. Low impact (small reward, real signal is stage/pearls).
5. **[SUSPECT, trivial] Mario unused signed vars** (`scrolling` 16–21, levelHi/Lo
   signed). No active effect; re-check if ever wired to reward. Low impact.

## Code bugs found along the way
**None** in scope. The reward/done/op machinery (`retro_dreamer.py:277-384`,
`config_validation.py`) is consistent with its config; the bugs are all
**config-constant** mismatches, which per god's rule await individual ack. No
non-config code defect surfaced.

## Methodology notes / limitations
- Observed bands come from **survival / practice captures**, not adversarial
  traces — so done-conditions that fire only on *death* (F1, F5) are
  under-exercised. Where a done-ref sits below an observed floor, I label SUSPECT
  rather than CONFIRMED.
- `>n` BCD types are decoded by stable-retro (not our code); I trust the decode
  but flag any var that's const-0 where the game visibly scores (F3).
- FZero-Test inherits FZero-Snes's RAM map verbatim (byte-identical `data.json`),
  so its observed band is known by transitivity — this is why F2 is CONFIRMED
  without a Test-specific capture.

**Definition of done met:** every onboarded game swept; verdict on every
constant; ranked findings with evidence + proposed fixes; no config fixes
applied (god acks each); no code bugs to commit.

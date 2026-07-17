# Pam Vision Dryrun — Little Mermaid (lm.npz)

## RAM Report Summary

**Source:** `backend/episode_report.py` on `lm.npz` (1400 steps)

### Roles
- `stage` — milestone (reward=100, major)
- `green_pearls_found`, `red_pearls_found`, `score` — objectives (accumulator, never increased)
- `roomPos` — progress (counter, climbs×3)
- `playerPage` — progress (counter, climbs×2)
- `scrollY` — progress (rewarded positional)
- `health` — resource (penalty=5)
- `lives` — resource, terminal (penalty=50, in done-condition)

### Key Events (21 total)
- Deaths at step 477 and step 988 (both at playerPage=6, roomPos=1)
- 7 damage events (health decrements)
- 3 life segments: 477 steps, 511 steps, 412 steps

### Post-Mortem Verdict
- **Outcome:** 2 deaths, 7 damage events over 1400 steps
- **Stage:** NEVER ADVANCED (stayed at 0)
- **Objectives:** 0 green pearls, 0 red pearls, 0 score collected
- **Failure location:** Both deaths at playerPage=6, roomPos=1 (scrollY=422 and 418)
- **Stall:** No progress after step 347 — 1053 of 1400 steps stalled
- **Done condition:** lives went to 0 but never negative; margin=0, strict boundary not met
- **Skeleton verdict:** Died 2×, looping 3×, stalled after early progress, never advanced stage, collected nothing

---

## Frame Observations

### Frame f0001.png (episode start / early life 1)
**Scene:** Underwater setting — dark blue/black ocean background with coral reef formations along both left and right edges (reddish-pink coral). Blue coral/sea-plants fill the upper and middle areas. A purple platform ledge occupies the lower-left quadrant with a small green conical object (possibly a collectible or plant) sitting on it. A pink/red character sprite (the mermaid protagonist) is visible mid-right, appearing to swim. The HUD in the top-left shows four heart icons (health = 4? — though the report says health starts at 3, so one heart may be the starting state indicator rather than count) and what appears to be a score/counter area. Overall composition: 8-bit underwater platformer, left-to-right progression implied by platform placement and character position.

### Frame f0005.png (later in life 1, same area)
**Scene:** Visually nearly identical to f0001 — same underwater coral reef environment, same purple platform, same green object, same HUD layout. The pink character sprite appears to have moved slightly (still mid-right area) and now has a purplish hue on the lower body (tail), suggesting a different animation frame or pose. The scene composition has not changed — the agent is still in the same area of the level. This supports the report's stall finding: the agent spends a lot of time in the same location without meaningful progress.

### Frame f0009.png (death zone area — frame ~9 of 12)
**Scene:** Dramatically different environment — open water with a light-to-dark blue vertical gradient background (sunlight rays penetrating from surface). No solid platforms visible; the character is swimming in open water. Pink/red hair with green tail is clearly visible on the character sprite (Ariel-like mermaid). Coral reef bed along the bottom edge in reddish-pink. HUD top-left shows five heart slots — four filled with solid magenta hearts, one appears to be an outline/dimmed heart (suggesting health = 4 out of 5, or possibly 3 if one is a max-marker). This appears to be a different room/area than frames 1–5 — consistent with the report's `roomPos` reaching max 5 and then looping back. The green conical object is also visible near the bottom, sitting among the coral.

### Frame f0012.png (end of death-zone sequence — final frame)
**Scene:** Nearly identical to f0009 — same open water scene with blue gradient and sunlight rays. The character sprite is in the same central position, same pose. The purple platform ledge is visible at the bottom-right (not present in f0009), and the green conical object sits among the coral near it. The HUD hearts appear identical to f0009 (four solid, one outline). The visual similarity between frames 9 and 12 reinforces the stall pattern: even in the death-zone area, the agent is not making visible progress — it's hovering in roughly the same spot across the final frames of the extracted sequence.

---

## Fused Diagnosis

### Claims (tagged by evidence source)

1. **[report]** The agent never advanced `stage` — it stayed at 0 for all 1400 steps.
2. **[report]** All objectives (green pearls, red pearls, score) remained at 0 — nothing was collected.
3. **[report]** The agent died twice (steps 477 and 988), both at playerPage=6, roomPos=1, with scrollY near 420.
4. **[report]** 1053 of 1400 steps (75%) occurred after the last progress milestone at step 347 — the agent stalled for the majority of the episode.
5. **[report]** The done condition (`lives < 0`) never fired — lives went to 0 but the strict boundary was not met (margin 0).
6. **[vision]** The game is an underwater 8-bit platformer set in a coral reef environment. The protagonist is a mermaid character (pink hair, green tail).
7. **[vision]** The HUD uses heart icons to display health — 5 slots, with some filled/dimmed depending on state.
8. **[vision]** Two distinct scene types exist: (a) enclosed coral-reef caves with purple platforms, and (b) open water with blue gradient and sunlight rays.
9. **[vision]** The green conical object appears in multiple scenes but was never collected — consistent with zero `green_pearls_found`.
10. **[vision]** The agent's character sprite appears to hover in roughly the same screen position across frames, confirming low movement activity.
11. **[inference]** The "death zone" at playerPage=6, roomPos=1 corresponds to the open-water area seen in frames 9 and 12 — the agent likely dies from environmental hazards (e.g., running out of air, being pushed into obstacles) in this open space.
12. **[inference]** The agent reaches the open-water area early (within life 1, by step 347) but cannot progress beyond it — it loops back to the same room repeatedly without collecting items or advancing the stage.
13. **[inference]** The stall is not due to the agent being stuck in one pixel — it loops through rooms (roomPos resets 3×) but fails to make *new* progress each cycle.

### Where Vision Added Value Over Report Alone

- **[vision]** Confirmed the game genre and visual structure — without this, the report's "stage never advanced" is abstract; now we know the agent is swimming in a coral reef game and never makes it past the open-water zone.
- **[vision]** The heart-HUD layout (5 slots) clarifies the `health` RAM semantics — max health is likely 3 (matching the report's health starting at 3), with the 5th slot possibly being a "max" indicator or unused.
- **[vision]** The green object visible but uncollected in frames matches the report's `green_pearls_found = 0`, giving a concrete visual anchor for what "collecting" would look like.
- **[inference]** The two scene types (cave vs. open water) help explain the roomPos cycling: the agent traverses from cave to open water and back, but never breaks through the open-water bottleneck.

### Where Vision Tempted Over-Claims

- I was tempted to say the health HUD shows "4/5 health" in frames 9–12, but the report shows health = 3 at start. The visual evidence is ambiguous — one heart may be a max-marker, a "reserved" slot, or the HUD may show lives + health combined. **Lesson: HUD icon counts can be misleading without ground-truth mapping.**
- I was tempted to infer the open water is the "death zone" hazard, but the frames were extracted from the death-zone vicinity, not necessarily *at* the moment of death. The agent could die from a different cause (e.g., contact with an enemy not captured in these frames).
- I noticed the green object in multiple scenes and called it a "pearl," but it could be decorative or uncollectible. The report only tracks `green_pearls_found` as a RAM value — it does not confirm this specific on-screen object is the one that increments it.

### Verdict

**Did vision add anything the report alone lacked?** Yes, but modestly. The report tells us *what* happened (stalled, died, collected nothing); vision tells us *where* it happened (open water zone after coral caves) and *what the game looks like* (8-bit underwater platformer, heart HUD, green collectible objects). The visual context makes the report's abstractions concrete but does not add new diagnostic depth for this specific failure mode.

**One concrete recommendation for product primer frame scoping:**

**2 frames are worth their cost — no more.** Specifically:
1. **One frame from the start of the episode** (like f0001) — establishes game genre, HUD layout, and initial scene. This is essential for grounding any primer that references "what the game looks like."
2. **One frame from the death zone or failure area** (like f0012) — shows where the agent consistently fails, grounding the report's "failure location" in visual reality.

Intermediate frames add diminishing returns: frames 5 and 9 in this test were near-duplicates of frames 1 and 12 respectively. The cost of encoding/viewing a frame on this server (~seconds per frame) is not justified beyond the start + failure-anchor pair.

**For the product primer:** Scope frame references to exactly 2 frames per episode (start + death/endpoint), describe only scene/color/composition and coarse HUD elements, and never rely on fine text or icon-count precision. Tag every vision-derived claim as `[vision]` to prevent contamination with `[report]` data.

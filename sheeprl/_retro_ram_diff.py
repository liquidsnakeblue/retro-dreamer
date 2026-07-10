"""RAM diffing: find addresses that encode a game event, from capture .npz
files (made by _retro_ram_capture.py) plus marked step indices.

Two modes, matching how events actually get found:

  boundary — one event step per capture ("the race ended at step 812"):
    candidates are bytes stable for `window` steps before the mark, stable
    at a DIFFERENT value after it, in EVERY capture. This is exactly the
    triple-capture intersect that found F-Zero's race_on byte.

  classes — step ranges labeled A vs B in one capture ("steps 100-800 are
    racing, 900-1200 are the menu"): bytes constant within each class and
    different between classes.

Last stdout line: RESULT {candidates: [{addr_hex, flat_index, before, after}...]}

Usage:
  python _retro_ram_diff.py boundary <window> <npz:event_step> [<npz:event_step> ...]
  python _retro_ram_diff.py classes <npz> <a_start-a_end> <b_start-b_end>
"""
import json
import sys

import numpy as np


def flat_to_addr(flat_idx, offsets, sizes):
    for off, size in zip(offsets, sizes):
        if flat_idx < size:
            return int(off) + int(flat_idx)
        flat_idx -= size
    return -1


def stable_value(block):
    """Return the constant value of each column, or -1 where not constant."""
    first = block[0].astype(np.int16)
    const = (block == block[0]).all(axis=0)
    return np.where(const, first, -1)


mode = sys.argv[1]
candidate_mask = None
before_vals = after_vals = None
offsets = sizes = None

if mode == "boundary":
    window = int(sys.argv[2])
    for spec in sys.argv[3:]:
        path, ev = spec.rsplit(":", 1)
        ev = int(ev)
        d = np.load(path)
        ram, offsets, sizes = d["ram"], d["offsets"], d["sizes"]
        pre = stable_value(ram[max(0, ev - window):ev])
        post = stable_value(ram[ev:min(len(ram), ev + window)])
        mask = (pre >= 0) & (post >= 0) & (pre != post)
        if candidate_mask is None:
            candidate_mask, before_vals, after_vals = mask, pre, post
        else:
            # values must agree across captures too, not just "changed"
            mask &= (pre == before_vals) & (post == after_vals)
            candidate_mask &= mask
        print(f"{path} @ {ev}: {int(mask.sum())} boundary bytes "
              f"(running intersect: {int(candidate_mask.sum())})", flush=True)
elif mode == "classes":
    d = np.load(sys.argv[2])
    ram, offsets, sizes = d["ram"], d["offsets"], d["sizes"]
    a0, a1 = map(int, sys.argv[3].split("-"))
    b0, b1 = map(int, sys.argv[4].split("-"))
    va, vb = stable_value(ram[a0:a1]), stable_value(ram[b0:b1])
    candidate_mask = (va >= 0) & (vb >= 0) & (va != vb)
    before_vals, after_vals = va, vb
    print(f"class A steps {a0}-{a1} vs B {b0}-{b1}: {int(candidate_mask.sum())} bytes", flush=True)
else:
    raise SystemExit(f"unknown mode {mode}")

idxs = np.flatnonzero(candidate_mask)
cands = [
    {
        "addr_hex": hex(flat_to_addr(int(i), offsets, sizes)),
        "flat_index": int(i),
        "before": int(before_vals[i]),
        "after": int(after_vals[i]),
    }
    for i in idxs[:200]
]
for c in cands[:25]:
    print(f"  {c['addr_hex']}: {c['before']} -> {c['after']}", flush=True)
print("RESULT " + json.dumps({"n_candidates": len(idxs), "candidates": cands}), flush=True)

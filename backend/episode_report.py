#!/usr/bin/env python3
"""Game-AGNOSTIC episode report from a RAM-capture .npz + the game's training.json.

Decodes the game's own memory into an EVENT STREAM + a compact post-mortem, so a
model reasons over labeled facts instead of squinting at pixels.

Roles are derived from config (done/reward) + trajectory shape — no per-game code.
Usage: episode_report.py <capture.npz> <training.json>
"""
import sys, json
import numpy as np

NON_VAR = {"ram", "offsets", "sizes", "ckpt", "state"}


def load(npz_path, cfg_path):
    d = np.load(npz_path, allow_pickle=True)
    vars_ = {k: d[k].astype(float) for k in d.files if k not in NON_VAR and d[k].ndim == 1}
    cfg = json.load(open(cfg_path))
    return vars_, cfg, len(next(iter(vars_.values())))


def assign_roles(vars_, cfg):
    """Return {name: {'roles': set, 'wrap': int|None, 'why': str}} — game-agnostic.

    DECLARED roles (from done/reward config) win over trajectory, so a var that
    stays constant this episode (e.g. stage=0) still carries its true role and
    the post-mortem can report 'never advanced'.
    """
    rcfg = (cfg.get("reward") or {}).get("variables", {})
    dcfg = (cfg.get("done") or {}).get("variables", {})
    # Breadcrumb rewards reference vars OUTSIDE reward.variables — surface
    # them as declared roles so a never-firing milestone reads as "milestone
    # never reached", not as an undeclared constant.
    _mcfg = (cfg.get("reward") or {}).get("milestones", {}) or {}
    _ncfg = (cfg.get("reward") or {}).get("novelty", {}) or {}
    _ccfg = (cfg.get("reward") or {}).get("counters", {}) or {}
    ms_vars = {c.get("var") for c in _mcfg.values() if isinstance(c, dict)}
    nv_keys = {k for c in _ncfg.values() if isinstance(c, dict)
               for k in (c.get("keys") or [])}
    ct_vars = {c.get("var") for c in _ccfg.values() if isinstance(c, dict)}
    ct_ctx = {k for c in _ccfg.values() if isinstance(c, dict)
              for k in (c.get("context") or [])}
    roles = {}
    for name, arr in vars_.items():
        R, why = set(), []
        r = rcfg.get(name, {})
        wrap = r.get("wrap")
        finite = arr[np.isfinite(arr)]
        span = (finite.max() - finite.min()) if finite.size else 0
        monotonic_up = finite.size > 1 and np.all(np.diff(finite) >= 0)
        n_distinct = len(np.unique(finite))
        rises = _count_climbs(arr)
        timer = _is_countdown(arr)
        continuous = _is_continuous(arr)
        # 1) terminal — in the done condition
        if name in dcfg:
            R.add("terminal"); why.append("in done-condition")
        # 2) timer/resource — regular countdowns are clocks, not damage
        if timer:
            R.add("timer"); why.append("regular near-monotonic countdown")
        elif "penalty" in r:
            R.add("resource"); why.append(f"penalty={r['penalty']}")
        # 3) rewarded gains: milestone / objective / positional-progress
        if not timer and "reward" in r and "mode" not in r:
            rv = r.get("reward", 0)
            if rv >= 50:
                R.add("milestone"); why.append(f"reward={rv} (major)")
            elif wrap is not None:
                R.add("progress"); R.add("cyclic"); why.append(f"rewarded, wrap={wrap} (cyclic position)")
            elif monotonic_up:
                R.add("objective"); why.append("rewarded, only-increases (accumulator)")
            else:
                R.add("progress"); why.append("rewarded positional")
        # Breadcrumb-declared vars (one-shot milestones / novelty keys)
        if name in ms_vars:
            R.add("milestone"); why.append("milestone breadcrumb (one-shot)")
        if name in nv_keys:
            R.add("progress"); why.append("novelty visited-set key")
        if name in ct_vars:
            R.add("objective"); why.append("counter event source (diminishing)")
        if name in ct_ctx:
            R.add("context"); why.append("counter attribution context")
        # Mode-style and future reward configs are still declared reward
        # signals even when they do not expose a scalar delta-reward key.
        known = {"timer", "resource", "milestone", "objective", "progress"}
        if name in rcfg and not (R & known):
            R.add("rewarded")
            mode = f"mode={r['mode']}, " if r.get("mode") is not None else ""
            shape = "continuous signal" if continuous else "reward signal"
            why.append(f"{mode}{shape}")
        # 4) undeclared counter that climbs & resets -> progress (e.g. page/room)
        if not R and span > 0 and 1 < n_distinct <= 64 and rises >= 1:
            if rises > span + 1:
                R.add("context"); why.append(f"oscillator climbs×{rises} within span {span:g}")
            else:
                R.add("progress"); why.append(f"counter climbs×{rises} (unrewarded)")
        if not R and span > 0:
            R.add("context"); why.append("varies, no declared role")
        if not R:
            R.add("context"); why.append("constant this episode")
        roles[name] = {
            "roles": R, "wrap": wrap, "why": ", ".join(why),
            "continuous": continuous,
            "configured": (name in rcfg or name in ms_vars or name in nv_keys
                           or name in ct_vars or name in ct_ctx),
            "max_delta": r.get("max_delta"), "n_distinct": n_distinct,
        }
    return roles


def _count_climbs(arr, drop_frac=0.6):
    """How many times the value climbs to a local peak then resets low (a loop)."""
    finite = arr[np.isfinite(arr)]
    if finite.size < 3:
        return 0
    lo, hi = finite.min(), finite.max()
    if hi - lo == 0:
        return 0
    thresh_hi = lo + 0.8 * (hi - lo)
    thresh_lo = lo + 0.2 * (hi - lo)
    cycles, armed = 0, False
    for v in finite:
        if v >= thresh_hi:
            armed = True
        elif v <= thresh_lo and armed:
            cycles += 1; armed = False
    return cycles


def _shape_stats(arr):
    """Observed span/cardinality/typical move, ignoring missing and flat steps."""
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return 0, len(np.unique(finite)), 0
    moves = np.abs(np.diff(finite))
    moves = moves[moves > 0]
    return finite.max() - finite.min(), len(np.unique(finite)), \
        (float(np.median(moves)) if moves.size else 0)


def _is_continuous(arr):
    """Large-cardinality or wide-vs-typical-delta signal, not a small counter."""
    span, n_distinct, typical = _shape_stats(arr)
    return n_distinct > 64 or (
        n_distinct > 16 and typical > 0 and span >= 16 * typical
    )


def _is_countdown(arr):
    """Shape-only clock detector: mostly equal decrements at a regular cadence."""
    finite = arr[np.isfinite(arr)]
    if finite.size < 3:
        return False
    delta = np.diff(finite)
    changed = delta != 0
    downs = -delta[delta < 0]
    if downs.size < 8 or downs.size / max(1, changed.sum()) < 0.90:
        return False
    typical = float(np.median(downs))
    if np.mean(np.isclose(downs, typical, rtol=0.05, atol=1e-9)) < 0.90:
        return False
    down_at = np.flatnonzero(delta < 0)
    gaps = np.diff(down_at)
    if gaps.size == 0:
        return False
    _, counts = np.unique(gaps, return_counts=True)
    return counts.max() / gaps.size >= 0.75


def _cond_true(op, v, ref):
    """Boolean array: is the done condition met at each step? (game-agnostic ops)."""
    if op in ("less-than", "lt", "<"):
        return v < ref
    if op in ("greater-than", "gt", ">"):
        return v > ref
    if op in ("equal", "eq", "=="):
        return v == ref
    if op in ("less-equal", "le", "<="):
        return v <= ref
    if op in ("greater-equal", "ge", ">="):
        return v >= ref
    return np.zeros_like(v, dtype=bool)


def _life_vars(vars_, roles, cfg):
    """Small discrete penalized counters that participate in episode end."""
    dcfg = (cfg.get("done") or {}).get("variables", {})
    found = set()
    for name, meta in roles.items():
        if not ({"terminal", "resource"} <= meta["roles"]) or name not in dcfg:
            continue
        arr = np.asarray(vars_[name], dtype=float)
        cond = dcfg[name]
        met = _cond_true(cond.get("op"), arr, cond.get("reference"))
        hits = np.flatnonzero(met)
        # Inspect only through the first terminal hit: post-terminal underflow or
        # wrapping must not make a small counter look like a continuous gauge.
        sample = arr[:int(hits[0]) + 1] if hits.size else arr
        span, n_distinct, _ = _shape_stats(sample)
        if n_distinct <= 16 and span <= 10:
            found.add(name)
    return found


def _at_life_floor(cond, value):
    """Whether a prior value is already at/beyond a terminal counter boundary."""
    op, ref = cond.get("op"), cond.get("reference")
    if op in ("less-than", "lt", "<", "less-equal", "le", "<=",
              "equal", "eq", "=="):
        return value <= ref
    if op in ("greater-than", "gt", ">", "greater-equal", "ge", ">="):
        return value >= ref
    return False


def _resource_threshold(arr):
    """Cumulative movement needed for a significant continuous-gauge event."""
    span, _, typical = _shape_stats(arr)
    return max(1.0, 0.10 * span, 8.0 * typical)


def detect_events(vars_, roles, cfg, T):
    """Generic operators over roled variables -> chronological events.

    Terminal is detected by the done condition actually becoming TRUE
    (threshold crossing per its op/reference), not by a terminal variable move.
    """
    dcfg = (cfg.get("done") or {}).get("variables", {})
    life_vars = _life_vars(vars_, roles, cfg)
    ev = []
    for name, meta in roles.items():
        R, arr = meta["roles"], vars_[name]
        v = np.where(np.isfinite(arr), arr, np.nan)
        gmax, lo = np.nanmax(v), np.nanmin(v)
        continuous_resource = ("resource" in R and meta["continuous"]
                               and name not in life_vars)
        if continuous_resource:
            finite_at = np.flatnonzero(np.isfinite(v))
            if finite_at.size:
                anchor = v[finite_at[0]]
                threshold = _resource_threshold(v)
                for t in range(int(finite_at[0]) + 1, T):
                    b = v[t]
                    if not np.isfinite(b) or abs(b - anchor) < threshold:
                        continue
                    kind = "loss" if b < anchor else "regain"
                    delta = b - anchor
                    change = f"-{abs(delta):g}" if delta < 0 else f"+{delta:g}"
                    ev.append((t, kind, name,
                               f"{name} {anchor:g}->{b:g} ({change}, significant)"))
                    anchor = b
        seen_max = -np.inf
        for t in range(1, T):
            a, b = v[t - 1], v[t]
            if not (np.isfinite(a) and np.isfinite(b)) or a == b:
                continue
            below_floor = (name in life_vars and name in dcfg
                           and _at_life_floor(dcfg[name], a))
            if "resource" in R and not continuous_resource and b < a and not below_floor:
                ev.append((t, "loss", name, f"{name} {int(a)}->{int(b)} (-{int(a-b)})"))
            if "resource" in R and not continuous_resource and b > a and not below_floor:
                ev.append((t, "regain", name, f"{name} {int(a)}->{int(b)}"))
            if "objective" in R and b > a:
                ev.append((t, "objective+", name, f"{name} +{b-a:g} (={b:g})"))
            if ("progress" in R or "milestone" in R) and "cyclic" not in R:
                if b > seen_max and b >= gmax:
                    ev.append((t, "milestone", name, f"{name} reached {int(b)} (max)"))
                if b < a and a >= (lo + 0.8 * (gmax - lo)) and b <= (lo + 0.2 * (gmax - lo)):
                    ev.append((t, "reset/loop", name, f"{name} {int(a)}->{int(b)} (looped back)"))
            seen_max = max(seen_max, b)
    # terminal: each done condition crossing false -> true
    for var, cond in dcfg.items():
        if var not in vars_:
            continue
        v = np.where(np.isfinite(vars_[var]), vars_[var], np.nan)
        met = _cond_true(cond.get("op"), v, cond.get("reference"))
        for t in range(1, T):
            if met[t] and not met[t - 1]:
                if var in life_vars and _at_life_floor(cond, v[t - 1]):
                    continue
                ev.append((t, "TERMINAL", var, f"done: {var} {cond.get('op')} {cond.get('reference')}"))
    ev = _annotate_damage_locations(ev, vars_, roles)
    ev.sort(key=lambda e: e[0])
    return ev


def _dedup_deaths(events, life_vars, window=3):
    """Keep every counter loss; add only non-coincident terminal firings."""
    losses = [e for e in events if e[1] == "loss" and e[2] in life_vars]
    deaths = list(losses)
    for terminal in (e for e in events if e[1] == "TERMINAL"):
        if not any(abs(terminal[0] - prior[0]) <= window for prior in deaths):
            deaths.append(terminal)
    return sorted(deaths, key=lambda e: e[0])


def _trend_stats(arr):
    finite = np.asarray(arr, dtype=float)
    finite = finite[np.isfinite(finite)]
    width = max(1, int(0.20 * finite.size))
    return (finite.min(), finite.mean(), finite.max(), finite[-1],
            finite[:width].mean(), finite[-width:].mean())


def _unwrap_cyclic(arr, wrap, max_delta=None):
    """Convert a wrapped configured signal to relative movement."""
    v = np.asarray(arr, dtype=float)
    out = np.full(v.shape, np.nan)
    finite_at = np.flatnonzero(np.isfinite(v))
    if not finite_at.size:
        return out
    first = int(finite_at[0])
    out[first] = 0.0
    total, previous = 0.0, v[first]
    for t in range(first + 1, len(v)):
        if not np.isfinite(v[t]):
            continue
        delta = v[t] - previous
        if wrap:
            delta = (delta + wrap / 2) % wrap - wrap / 2
        if max_delta is not None and abs(delta) > max_delta:
            delta = 0.0
        total += delta
        out[t] = total
        previous = v[t]
    return out


def _axis_names(vars_, roles, include_constant=True):
    names = []
    for name, meta in roles.items():
        if not (meta["roles"] & {"progress", "milestone"}) or "cyclic" in meta["roles"]:
            continue
        finite = vars_[name][np.isfinite(vars_[name])]
        if finite.size and (include_constant or finite.max() != finite.min()):
            names.append(name)
    return sorted(names)


def _annotate_damage_locations(events, vars_, roles):
    """Attach a compact position snapshot to significant resource losses."""
    plain = _axis_names(vars_, roles, include_constant=False)
    cyclic = sorted(n for n, m in roles.items()
                    if {"progress", "cyclic"} <= m["roles"] and m["configured"])
    chosen = plain if plain else cyclic
    tracks = {}
    for name in chosen:
        meta = roles[name]
        tracks[name] = (_unwrap_cyclic(vars_[name], meta["wrap"], meta["max_delta"])
                        if "cyclic" in meta["roles"] else vars_[name])
    annotated = []
    for t, kind, name, desc in events:
        significant_loss = (kind == "loss" and name in roles
                            and "resource" in roles[name]["roles"]
                            and roles[name]["continuous"])
        coords = []
        if significant_loss:
            for axis in chosen:
                if axis == name or not 0 <= t < len(tracks[axis]):
                    continue
                value = tracks[axis][t]
                if not np.isfinite(value):
                    continue
                value = 0.0 if abs(value) < 1e-12 else value
                suffix = " rel" if "cyclic" in roles[axis]["roles"] else ""
                sign = "+" if suffix else ""
                coords.append(f"{axis}={value:{sign}g}{suffix}")
        annotated.append((t, kind, name,
                          desc + (f" @ {', '.join(coords)}" if coords else "")))
    return annotated


def _done_margin_lines(vars_, cfg):
    """Final value and operator-aware closest approach for every done input."""
    dcfg = (cfg.get("done") or {}).get("variables", {})
    lower = {"less-than", "lt", "<", "less-equal", "le", "<="}
    upper = {"greater-than", "gt", ">", "greater-equal", "ge", ">="}
    equal = {"equal", "eq", "=="}
    strict = {"less-than", "lt", "<", "greater-than", "gt", ">"}
    symbols = {**{op: "<" for op in ("less-than", "lt", "<")},
               **{op: "<=" for op in ("less-equal", "le", "<=")},
               **{op: ">" for op in ("greater-than", "gt", ">")},
               **{op: ">=" for op in ("greater-equal", "ge", ">=")},
               **{op: "==" for op in equal}}
    lines = []
    for name, cond in dcfg.items():
        if name not in vars_:
            lines.append(f"  done margin · {name}: unavailable (not captured)")
            continue
        arr = np.asarray(vars_[name], dtype=float)
        finite_at = np.flatnonzero(np.isfinite(arr))
        if not finite_at.size:
            lines.append(f"  done margin · {name}: unavailable (no finite samples)")
            continue
        op, ref = cond.get("op"), cond.get("reference")
        if op not in symbols or ref is None:
            lines.append(f"  done margin · {name}: unsupported condition {op} {ref}")
            continue
        finite = arr[finite_at]
        if op in lower:
            local = int(np.argmin(finite)); approach = "low"
        elif op in upper:
            local = int(np.argmax(finite)); approach = "high"
        else:
            local = int(np.argmin(np.abs(finite - ref))); approach = "nearest"
        closest = finite[local]
        met_at = np.flatnonzero(_cond_true(op, arr, ref))
        if met_at.size:
            status = f"met @step {int(met_at[0])}"
        else:
            gap = abs(closest - ref)
            boundary = "strict boundary; " if op in strict and np.isclose(gap, 0) else ""
            noun = "distance" if op in equal else "margin"
            status = f"not met ({boundary}{noun} {gap:g})"
        final_at = int(finite_at[-1])
        final = f"final {arr[final_at]:g}"
        if final_at != len(arr) - 1:
            final = f"last observed {arr[final_at]:g} @step {final_at}"
        lines.append(f"  done margin · {name}: {final}; {approach} {closest:g} "
                     f"vs {symbols[op]} {ref:g} — {status}")
    return lines


def _death_snapshot_lines(vars_, axis, deaths):
    lines = []
    for t, *_ in deaths:
        coords = [f"{name}={vars_[name][t]:g}" for name in axis
                  if 0 <= t < len(vars_[name]) and np.isfinite(vars_[name][t])]
        lines.append(f"  death @{t}: " + (", ".join(coords) if coords else "no progress axis"))
    return lines


def _life_segment_lines(vars_, axis, deaths, terminals, T, window=3):
    """Half-open retry spans; captured terminal tails are not new segments."""
    cuts = sorted({int(e[0]) for e in deaths if 0 < e[0] < T})
    if not cuts:
        return []
    terminal_at = [int(e[0]) for e in terminals]
    terminal_cut = next((t for t in cuts
                         if any(abs(t - end) <= window for end in terminal_at)), None)
    if terminal_cut is not None:
        cuts = [t for t in cuts if t <= terminal_cut]
        bounds = [0] + cuts
    else:
        bounds = [0] + cuts + [T]
    lines = ["  LIFE SEGMENTS:"]
    for number, (start, end) in enumerate(zip(bounds, bounds[1:]), 1):
        starts, maxima = [], []
        for name in axis:
            segment = vars_[name][start:end]
            finite = segment[np.isfinite(segment)]
            if not finite.size:
                continue
            start_value = vars_[name][start]
            if np.isfinite(start_value):
                starts.append(f"{name}={start_value:g}")
            maxima.append(f"{name}={finite.max():g}")
        detail = ""
        if starts or maxima:
            detail = f" · start {', '.join(starts)}; max {', '.join(maxima)}"
        lines.append(f"    life {number} · steps {start}–{end} ({end-start}){detail}")
    if terminal_cut is not None and terminal_cut < T:
        lines.append(f"    post-terminal tail · steps {terminal_cut}–{T} "
                     f"({T-terminal_cut}) captured after done")
    return lines


def summarize(vars_, roles, events, cfg, T):
    # the progress AXIS excludes cyclic (wrap) positional detail
    axis = _axis_names(vars_, roles)
    active_axis = _axis_names(vars_, roles, include_constant=False)
    obj = [n for n, m in roles.items() if "objective" in m["roles"]]

    def const(n):
        v = vars_[n]; return np.nanmax(v) == np.nanmin(v)
    life_vars = _life_vars(vars_, roles, cfg)
    # Counter losses are deaths; coincident done-condition firings are evidence,
    # not additional deaths.
    deaths = _dedup_deaths(events, life_vars)
    damage = [e for e in events if e[1] == "loss" and e[2] not in life_vars]
    terminals = [e for e in events if e[1] == "TERMINAL"]

    lines = []
    out = f"{len(deaths)} death/fail" if deaths else "survived the window (no death/terminal)"
    extra = []
    if damage:
        damage_vars = {e[2] for e in damage}
        adjective = "significant " if all(roles[n]["continuous"] for n in damage_vars) else ""
        extra.append(f"{len(damage)} {adjective}damage events")
    if terminals:
        extra.append(f"done-condition fired ×{len(terminals)}")
    lines.append(f"OUTCOME: {out}" + (f" · {', '.join(extra)}" if extra else "") + f" over {T} steps.")
    # reach — furthest each progress axis got, and where it stalled
    for n in sorted(axis):
        v = vars_[n]; gm = np.nanmax(v)
        if const(n):
            lines.append(f"  reach · {n}: NEVER ADVANCED (stayed {int(gm)})")
        else:
            lines.append(f"  reach · {n}: max {int(gm)} first hit @step {int(np.nanargmax(v))} (last {int(v[-1])})")
    # objectives — report every declaration, including empty accumulators
    for n in sorted(obj):
        lines.append(f"  objective · {n}: {int(np.nanmax(vars_[n]))} gained")
    if not obj:
        lines.append("  objective · none declared")
    # Clocks and configured high-information signals get compact trajectory
    # summaries instead of event-per-tick narration.
    for n, m in sorted(roles.items()):
        if "timer" not in m["roles"]:
            continue
        v = vars_[n]
        finite = v[np.isfinite(v)]
        resets = int(np.sum(np.diff(finite) > 0))
        reset_text = f"; resets ×{resets}" if resets else ""
        lines.append(f"  timer · {n}: start {finite[0]:g}, low {finite.min():g}, "
                     f"end {finite[-1]:g}{reset_text}")
    for n, m in sorted(roles.items()):
        if not (m["configured"] and m["continuous"] and "resource" in m["roles"]):
            continue
        lo, mean, hi, final, early, late = _trend_stats(vars_[n])
        lines.append(f"  resource · {n}: min {lo:g}, mean {mean:.1f}, max {hi:g}, "
                     f"final {final:g}; trend {early:.1f}→{late:.1f}")
    signal_names = []
    for n, m in sorted(roles.items()):
        if not (m["configured"] and m["continuous"]
                and m["roles"] & {"context", "rewarded"}):
            continue
        lo, mean, hi, _, early, late = _trend_stats(vars_[n])
        lines.append(f"  signal · {n}: min {lo:g}, mean {mean:.1f}, max {hi:g}; "
                     f"trend {early:.1f}→{late:.1f}")
        signal_names.append(n)
    for n, m in sorted(roles.items()):
        if not (m["configured"] and "cyclic" in m["roles"]
                and 1 < m["n_distinct"] <= 64):
            continue
        relative = _unwrap_cyclic(vars_[n], m["wrap"], m["max_delta"])
        lo, mean, hi, final, early, late = _trend_stats(relative)
        lines.append(f"  cyclic signal · {n}: relative min {lo:g}, mean {mean:.1f}, "
                     f"max {hi:g}; trend {early:.1f}→{late:.1f}, net {final:+g}")
        signal_names.append(n)
    lines.extend(_done_margin_lines(vars_, cfg))
    lines.extend(_death_snapshot_lines(vars_, active_axis, deaths))
    lines.extend(_life_segment_lines(vars_, active_axis, deaths, terminals, T))
    # modal failure location (non-cyclic progress axis, co-located at deaths)
    if deaths and axis:
        loc = "; ".join(f"{n}={_modal([int(vars_[n][t]) for t, *_ in deaths])}"
                        for n in sorted(axis) if not const(n))
        if loc:
            lines.append(f"  FAILURE LOCATION (at deaths): {loc}")
    # stall — last new high on any advancing progress axis
    last_prog = 0
    for n in axis:
        v = vars_[n]
        if const(n):
            continue
        idx = np.where(v >= np.nanmax(v))[0]
        if idx.size:
            last_prog = max(last_prog, int(idx[0]))
    if last_prog and last_prog < 0.7 * T:
        lines.append(f"  STALL: no progress axis set a new high after step {last_prog} "
                     f"({T-last_prog} of {T} steps stalled).")
    # verdict skeleton — loop count = the most any single progress axis reset
    # (near-simultaneous resets across vars are one loop, not several)
    per_axis_resets = {}
    for _, kind, name, _ in events:
        if kind == "reset/loop":
            per_axis_resets[name] = per_axis_resets.get(name, 0) + 1
    loops = max(per_axis_resets.values()) if per_axis_resets else 0
    never = [n for n in axis if const(n)]
    verdict = []
    if deaths:
        verdict.append(f"died {len(deaths)}×")
    if loops >= 2:
        verdict.append(f"looping (×{loops})")
    if last_prog and last_prog < 0.7 * T:
        verdict.append("stalled after early progress")
    if never:
        verdict.append(f"never advanced {'/'.join(sorted(never))}")
    if obj and all(np.nanmax(vars_[n]) == 0 for n in obj):
        verdict.append("collected nothing")
    if not verdict and signal_names:
        verdict.append(f"survived with active {'/'.join(sorted(set(signal_names)))} signals")
    lines.append(f"VERDICT (skeleton): {', '.join(verdict) if verdict else 'progressing'}.")
    return "\n".join(lines)


def _modal(xs):
    vals, counts = np.unique(xs, return_counts=True)
    m = vals[counts == counts.max()].max()
    return f"{int(m)} ({counts.max()}/{len(xs)})"


def main():
    npz, cfg_path = sys.argv[1], sys.argv[2]
    vars_, cfg, T = load(npz, cfg_path)
    roles = assign_roles(vars_, cfg)
    events = detect_events(vars_, roles, cfg, T)

    print("=" * 68)
    print(f"EPISODE REPORT · {npz.split('/')[-1]} · {T} steps")
    print("=" * 68)
    print("\nROLES (derived from config + trajectory, no per-game code):")
    for n, m in sorted(roles.items(), key=lambda x: sorted(x[1]["roles"])):
        print(f"  {n:20} {','.join(sorted(m['roles'])):22} [{m['why']}]")
    print(f"\nEVENT STREAM ({len(events)} events):")
    for t, kind, name, desc in events:
        print(f"  step {t:5}  {kind:14} {desc}")
    print("\nPOST-MORTEM:")
    print(summarize(vars_, roles, events, cfg, T))


if __name__ == "__main__":
    main()

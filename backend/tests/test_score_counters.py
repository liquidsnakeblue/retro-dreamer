"""score_counters unit tests, driven by RAM traces recorded in-emulator on
Zelda 1 (Sol's 0x627 verification session, 2026-07-17). The byte is a
no-damage kill STREAK: +1 per enemy death, resets to 0 on the damage frame,
does NOT reset on screen transitions (stale value walks onto new screens),
and the screen-id byte flickers during scroll animations. Every trace below
reproduces one of those observed behaviors.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "cv_mod",
    Path(__file__).resolve().parent.parent.parent
    / "sheeprl" / "sheeprl" / "envs" / "config_validation.py",
)
_cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cv)
score_counters = _cv.score_counters
validate_training_config = _cv.validate_training_config

RULE = {
    "screen_kills": {
        "var": "kills", "context": ["level", "screen_id"],
        "reward": 2.0, "decay": 0.5, "max_per_context": 6,
        "max_event_delta": 3,
    }
}


def run(trace, cfg=RULE):
    """trace = list of info dicts (agent-step samples). Returns per-step pay."""
    state, pays = {}, []
    prev = trace[0]
    for cur in trace[1:]:
        pays.append(score_counters(cfg, prev, cur, state))
        prev = cur
    return pays, state


def I(level, screen, kills):
    return {"level": level, "screen_id": screen, "kills": kills}


def test_kills_pay_with_diminishing_returns():
    # Sol trace: three sword kills on screen 0x67 (settled, no damage)
    trace = [I(0, 0x67, 0), I(0, 0x67, 1), I(0, 0x67, 1), I(0, 0x67, 2), I(0, 0x67, 3)]
    pays, _ = run(trace)
    assert pays == [2.0, 0.0, 1.0, 0.5]  # 2 * 0.5^n


def test_cap_stops_payment():
    trace = [I(0, 0x67, k) for k in range(0, 9)]
    pays, _ = run(trace)
    assert sum(1 for p in pays if p > 0) == 6  # max_per_context
    assert pays[6] == 0.0 and pays[7] == 0.0


def test_damage_reset_is_not_an_event_and_cannot_rearm():
    # Sol trace: kills 3 -> 0 on the damage frame, then re-kills climb again
    trace = [I(0, 0x67, 2), I(0, 0x67, 3), I(0, 0x67, 0),
             I(0, 0x67, 1), I(0, 0x67, 2), I(0, 0x67, 3)]
    pays, state = run(trace)
    assert pays[0] == 2.0        # 3rd observed kill... first paid ordinal here
    assert pays[1] == 0.0        # damage reset: decrease ignored
    # post-damage kills are NEW events, paid at the NEXT ordinals (1,2,3)
    assert pays[2:] == [2.0 * 0.5, 2.0 * 0.25, 2.0 * 0.125]
    assert state["screen_kills"][(0, 0x67)] == 4


def test_stale_streak_on_new_screen_pays_nothing():
    # Sol trace: kills=3 carried across a screen transition (byte does NOT reset)
    trace = [I(0, 0x67, 3), I(0, 0x77, 3), I(0, 0x77, 3)]
    pays, state = run(trace)
    assert pays == [0.0, 0.0]    # arrival is not an event; no increment on 0x77
    assert state == {}


def test_scroll_flicker_cannot_pay():
    # Sol trace: screen byte flickers 0x67->0x77->0x67->0x77 during scrolling
    trace = [I(0, 0x67, 3), I(0, 0x77, 3), I(0, 0x67, 3), I(0, 0x77, 3)]
    pays, _ = run(trace)
    assert pays == [0.0, 0.0, 0.0]


def test_garbage_jump_rejected():
    trace = [I(0, 0x67, 0), I(0, 0x67, 203)]
    pays, _ = run(trace)
    assert pays == [0.0]         # exceeds max_event_delta


def test_multi_kill_delta_pays_each_event():
    # bomb double-kill in one agent step
    trace = [I(0, 0x67, 0), I(0, 0x67, 2)]
    pays, _ = run(trace)
    assert pays == [2.0 + 1.0]


def test_contexts_are_independent():
    trace = [I(0, 0x67, 0), I(0, 0x67, 1), I(0, 0x77, 1),  # move (stale byte)
             I(0, 0x77, 2)]                                 # kill on new screen
    pays, _ = run(trace)
    assert pays == [2.0, 0.0, 2.0]  # new context starts at decay^0


def test_flicker_with_simultaneous_increment_is_suppressed():
    # A kill increment landing on the exact transition/flicker frame is
    # deliberately NOT paid (conservative: never pay on a context boundary).
    # Locked down so nobody "fixes" this into stale-counter payouts.
    trace = [I(0, 0x67, 2), I(0, 0x77, 3), I(0, 0x67, 4), I(0, 0x67, 4)]
    pays, state = run(trace)
    assert pays == [0.0, 0.0, 0.0]
    assert state == {}


def test_decay_one_flat_until_cap():
    cfg = {"c": {"var": "kills", "context": ["level", "screen_id"],
                 "reward": 5.0, "decay": 1.0, "max_per_context": 3,
                 "max_event_delta": 1}}
    trace = [I(0, 1, k) for k in range(0, 6)]
    pays, _ = run(trace, cfg)
    assert pays == [5.0, 5.0, 5.0, 0.0, 0.0]


def test_missing_vars_are_safe():
    trace = [{"level": 0}, {"level": 0}]
    pays, _ = run(trace)
    assert pays == [0.0]


def test_validation_accepts_good_and_rejects_bad():
    good = {"reward": {"counters": RULE}}
    validate_training_config("t", good, data_vars={"kills", "level", "screen_id"})
    for bad, why in [
        ({"reward": {"counters": {"c": {"var": "kills", "context": ["level"], "reward": 2}}}},
         "missing max_per_context"),
        ({"reward": {"counters": {"c": {"var": "kills", "context": [], "reward": 2, "max_per_context": 5}}}},
         "empty context"),
        ({"reward": {"counters": {"c": {"var": "kills", "context": ["level"], "reward": 2, "max_per_context": 5, "decay": 1.5}}}},
         "decay out of range"),
        ({"reward": {"counters": {"c": {"var": "killz", "context": ["level"], "reward": 2, "max_per_context": 5}}}},
         "typo var"),
        ({"reward": {"counters": {"c": {"var": "kills", "context": ["level"], "reward": 2, "max_per_context": 5, "max_event_delta": 0}}}},
         "max_event_delta zero"),
        ({"reward": {"counters": {"c": {"var": "kills", "context": ["level"], "reward": "x", "max_per_context": 5}}}},
         "non-numeric reward"),
        ({"reward": {"counters": {"c": {"var": "kills", "context": ["level"], "reward": 2, "max_per_context": True}}}},
         "bool cap"),
        ({"reward": {"counters": {"c": {"var": None, "context": ["level"], "reward": 2, "max_per_context": 5}}}},
         "null var"),
    ]:
        try:
            validate_training_config("t", bad, data_vars={"kills", "level", "screen_id"})
            raise AssertionError(f"accepted bad config: {why}")
        except ValueError:
            pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")

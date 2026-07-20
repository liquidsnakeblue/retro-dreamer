"""check_done unit tests — success-vs-failure episode-end classification.

Context (F-Zero lap-line hesitation, 2026-07-19): race finish (race_on==0)
used to be emitted as terminated=True — the continue predictor learned
"finish line = death" and, since laps are visually identical at 64x64,
smeared braking-fear across EVERY lap approach (measured 8-11% speed dip).
Fix: done conditions may carry "success": true, and the wrapper emits those
as truncated (value bootstraps). check_done returns the FIRST match in
config order so failures listed first win same-step ties.
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
check_done = _cv.check_done
episode_end_flags = _cv.episode_end_flags
validate_training_config = _cv.validate_training_config

# Live F-Zero shape: failures (health, reverse) BEFORE success (race_on)
DONE = {
    "health": {"op": "less-than", "reference": 2048},
    "reverse": {"op": "equal", "reference": 1},
    "race_on": {"op": "equal", "reference": 0, "success": True},
}


def I(health=2048, reverse=0, race_on=1):
    return {"health": health, "reverse": reverse, "race_on": race_on}


def test_no_condition_returns_none():
    assert check_done(DONE, I()) is None


def test_failure_matches_by_name():
    assert check_done(DONE, I(health=1900)) == "health"
    assert check_done(DONE, I(reverse=1)) == "reverse"


def test_success_matches_by_name():
    assert check_done(DONE, I(race_on=0)) == "race_on"
    assert DONE["race_on"].get("success") is True


def test_same_step_tie_prefers_failure():
    # A crash on the exact finish frame must classify as a death, not a
    # success — config order (failures first) is the tiebreak. Locked down
    # so nobody reorders the dict and silently flips crash semantics.
    assert check_done(DONE, I(health=100, race_on=0)) == "health"


def test_missing_vars_are_safe():
    assert check_done(DONE, {"speed": 3000}) is None


def test_empty_config_safe():
    assert check_done({}, I()) is None
    assert check_done(None, I()) is None


def test_flags_pass_through_when_nothing_matches():
    # Inner-env flags (e.g. TimeLimit truncation from the frame-skip loop)
    # survive untouched when no done condition fires.
    assert episode_end_flags(DONE, I(), False, False) == (False, False)
    assert episode_end_flags(DONE, I(), False, True) == (False, True)


def test_failure_sets_terminated():
    assert episode_end_flags(DONE, I(health=100), False, False) == (True, False)
    # Failure does not clear an inner-env truncation
    assert episode_end_flags(DONE, I(health=100), False, True) == (True, True)


def test_success_routes_to_truncated_and_downgrades_inner_terminal():
    # THE lap-line fix, at the exact production routing: success end ->
    # truncated=True AND terminated forced False, even if the inner env
    # (scenario.json-era done) raised terminated for the same frame. This is
    # the test whose absence let the original 'finish = death' bug ship.
    assert episode_end_flags(DONE, I(race_on=0), False, False) == (False, True)
    assert episode_end_flags(DONE, I(race_on=0), True, False) == (False, True)


def test_same_step_crash_at_finish_is_still_a_death():
    # WL2-pit lesson generalized: failure conditions listed first win the
    # tie, so a crash frame that also satisfies a success var terminates.
    assert episode_end_flags(DONE, I(health=100, race_on=0), False, False) == (True, False)


def test_greater_than_op_supported():
    cfg = {"lap": {"op": "greater-than", "reference": 4}}
    assert check_done(cfg, {"lap": 5}) == "lap"
    assert check_done(cfg, {"lap": 4}) is None


def test_validation_accepts_success_flag_and_rejects_nonbool():
    good = {
        "reward": {"variables": {"pos": {"reward": 1}}},
        "done": {"variables": {"race_on": {"op": "==", "reference": 0, "success": True}}},
    }
    validate_training_config("t", good, data_vars={"pos", "race_on"})
    bad = {
        "reward": {"variables": {"pos": {"reward": 1}}},
        "done": {"variables": {"race_on": {"op": "==", "reference": 0, "success": "yes"}}},
    }
    try:
        validate_training_config("t", bad, data_vars={"pos", "race_on"})
        raise AssertionError("accepted non-bool success flag")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")

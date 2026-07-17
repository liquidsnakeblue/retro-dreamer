"""score_milestones unit tests — spawn-armed suppression semantics.

Context (Zelda 1 rotation curriculum, 2026-07-17): rotation states saved
PAST an objective (PostSwordExit spawns with sword=1) must not pay the
milestone lump every episode. The wrapper runs the episode's first reward
step as a baseline pass (baseline=True): condition-true milestones are
consumed unpaid; condition-false milestones stay armed and pay normally on
a later genuine transition. stable-retro reset() info lacks data.json vars,
which is WHY baselining happens on the first step rather than at reset.
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
score_milestones = _cv.score_milestones

# Live Zelda milestone config shape
MS = {
    "got_sword": {"var": "sword", "op": ">", "reference": 0, "reward": 50.0},
    "entered_dungeon1": {"var": "level", "op": ">", "reference": 0, "reward": 100.0},
    "died": {"var": "game_mode", "op": "==", "reference": 17, "reward": -10.0},
}


def I(sword=0, level=0, game_mode=5):
    return {"sword": sword, "level": level, "game_mode": game_mode}


def run_episode(trace, cfg=MS):
    """trace = list of info dicts, one per agent step (step 1 first).
    Mirrors the wrapper: first step is the baseline pass. Returns per-step pay."""
    fired, pays = set(), []
    for i, info in enumerate(trace):
        pays.append(score_milestones(cfg, info, fired, baseline=(i == 0)))
    return pays, fired


def test_spawn_armed_milestone_suppressed():
    # PostSwordExit spawn: sword already 1 at the first look. Consumed unpaid,
    # and never pays later in the episode.
    trace = [I(sword=1), I(sword=1), I(sword=1)]
    pays, fired = run_episode(trace)
    assert pays == [0.0, 0.0, 0.0]
    assert "got_sword" in fired


def test_milestone_after_baseline_pays():
    # Condition false at baseline -> stays armed -> pays on the real transition.
    trace = [I(sword=0), I(sword=0), I(sword=1), I(sword=1)]
    pays, _ = run_episode(trace)
    assert pays == [0.0, 0.0, 50.0, 0.0]


def test_milestone_earned_within_first_action_window_is_suppressed():
    # A milestone earned DURING the very first agent action (frame_skip
    # window) is indistinguishable from a spawn-armed one — info first shows
    # the var AFTER that action. It is consumed unpaid. Deliberate, documented
    # cost of first-step baselining (see score_milestones docstring); locked
    # down so the tradeoff stays explicit and the name doesn't overpromise.
    trace = [I(sword=1), I(sword=1)]
    pays, fired = run_episode(trace)
    assert pays == [0.0, 0.0]
    assert "got_sword" in fired


def test_negative_milestone_also_baselined():
    # A state saved mid-death-cutscene must not charge -10 for a death the
    # agent didn't cause; a real death later still charges once.
    trace = [I(game_mode=17), I(game_mode=17), I(game_mode=5)]
    pays, fired = run_episode(trace)
    assert pays == [0.0, 0.0, 0.0]
    assert "died" in fired


def test_death_after_clean_spawn_charges_once():
    trace = [I(), I(), I(game_mode=17), I(game_mode=17)]
    pays, _ = run_episode(trace)
    assert pays == [0.0, 0.0, -10.0, 0.0]


def test_independent_milestones_mixed_baseline():
    # PostSwordExit spawn: got_sword suppressed but entered_dungeon1 stays
    # armed and pays when the agent genuinely reaches D1.
    trace = [I(sword=1), I(sword=1, level=1)]
    pays, fired = run_episode(trace)
    assert pays == [0.0, 100.0]
    assert fired == {"got_sword", "entered_dungeon1"}


def test_rotation_rebaseline_per_episode():
    # Episode on PostSwordExit (suppressed), then a FRESH episode on Overworld
    # (sword=0 spawn) earns the sword and is paid — reset() clears the set.
    pays1, _ = run_episode([I(sword=1), I(sword=1)])
    assert sum(pays1) == 0.0
    pays2, _ = run_episode([I(sword=0), I(sword=1)])
    assert pays2 == [0.0, 50.0]


def test_missing_var_safe_and_armed_later():
    # Var absent at baseline (e.g. partial info): milestone is NOT consumed,
    # and pays when the var appears and the condition is genuinely true...
    trace = [{"level": 0, "game_mode": 5}, I(sword=1)]
    pays, _ = run_episode(trace)
    # ...which is the documented cost: an armed spawn whose var was hidden at
    # baseline pays at first sight. Locked down so the tradeoff is explicit.
    assert pays == [0.0, 50.0]


def test_already_fired_skipped():
    fired = {"got_sword"}
    pay = score_milestones(MS, I(sword=1), fired, baseline=False)
    assert pay == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")

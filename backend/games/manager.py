"""GameManager — game discovery merging built-in stable-retro + custom games."""

import json
import os
import re
from pathlib import Path
from typing import Optional


VALID_CONFIG_FILENAMES = {
    "data.json",
    "scenario.json",
    "training.json",
    "actions.json",
    "metadata.json",
}

# TRUE emulator button order per core (stable_retro/cores/*.json) — action
# arrays index into these positions, so order and length must match exactly.
# "" marks an unused slot (the core's None placeholder); arrays still include it.
_SYSTEM_BUTTONS = {
    "Snes": ["B", "Y", "Select", "Start", "Up", "Down", "Left", "Right", "A", "X", "L", "R"],
    "Genesis": ["B", "A", "Mode", "Start", "Up", "Down", "Left", "Right", "C", "Y", "X", "Z"],
    "Nes": ["B", "", "Select", "Start", "Up", "Down", "Left", "Right", "A"],
    "GameBoy": ["B", "", "Select", "Start", "Up", "Down", "Left", "Right", "A"],
    "GbColor": ["B", "", "Select", "Start", "Up", "Down", "Left", "Right", "A"],
    "GbAdvance": ["B", "", "Select", "Start", "Up", "Down", "Left", "Right", "A", "", "L", "R"],
    "Gba": ["B", "", "Select", "Start", "Up", "Down", "Left", "Right", "A", "", "L", "R"],
    "Atari2600": ["Button", "", "Select", "Reset", "Up", "Down", "Left", "Right"],
    "Sms": ["B", "", "", "Pause", "Up", "Down", "Left", "Right", "A"],
    "GameGear": ["B", "", "", "Start", "Up", "Down", "Left", "Right", "A"],
    "PCEngine": ["II", "III", "Select", "Run", "Up", "Down", "Left", "Right", "I", "IV", "V", "VI"],
    "32x": ["B", "A", "Mode", "Start", "Up", "Down", "Left", "Right", "C", "Y", "X", "Z"],
    "Scd": ["B", "A", "Mode", "Start", "Up", "Down", "Left", "Right", "C", "Y", "X", "Z"],
    "Saturn": ["B", "A", "Mode", "Start", "Up", "Down", "Left", "Right", "C", "Y", "X", "Z"],
}


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_config_validation_mod = None


def _config_validation():
    """Load the shared config validators by FILE PATH.

    A plain `import sheeprl...` breaks in the server process: it runs from
    the repo root, where the outer ./sheeprl directory shadows the editable-
    installed package as a namespace package. The module is dependency-free,
    so loading its file directly sidesteps package resolution entirely.
    """
    global _config_validation_mod
    if _config_validation_mod is None:
        import importlib.util
        path = _PROJECT_ROOT / "sheeprl" / "sheeprl" / "envs" / "config_validation.py"
        spec = importlib.util.spec_from_file_location("_rd_config_validation", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _config_validation_mod = mod
    return _config_validation_mod


def _base_game_id(game_id: str) -> str:
    """Strip the integration version suffix built-in ids carry: '1942-Nes-v0' -> '1942-Nes'."""
    return re.sub(r"-v\d+$", "", game_id)


def _system_from_game_id(game_id: str) -> str:
    """Extract system: 'SonicTheHedgehog-Genesis' -> 'Genesis', '1942-Nes-v0' -> 'Nes'."""
    base = _base_game_id(game_id)
    if "-" in base:
        return base.rsplit("-", 1)[-1]
    return ""


def _display_name(game_id: str) -> str:
    """'SonicTheHedgehog-Genesis' -> 'Sonic The Hedgehog (Genesis)'."""
    base = _base_game_id(game_id)
    name_part = base.rsplit("-", 1)[0] if "-" in base else base
    display = ""
    for i, ch in enumerate(name_part):
        if ch.isupper() and i > 0 and name_part[i - 1].islower():
            display += " "
        display += ch
    system = _system_from_game_id(game_id)
    return f"{display} ({system})" if system else display


def _buttons_for_system(system: str) -> list[str]:
    return _SYSTEM_BUTTONS.get(system, [])


class GameManager:
    """Discovers games from both stable-retro's built-in library and custom games/ dir.

    Built-in games (1000+) come pre-configured with data.json, scenario.json, states.
    Custom games in games_dir can override or extend built-in ones with:
      - training.json  (custom reward shaping — our extension)
      - actions.json   (discrete action combos — our extension)
      - metadata.json, data.json, scenario.json overrides
    """

    def __init__(self, games_dir: Path):
        self.games_dir = Path(games_dir)
        self.games_dir.mkdir(parents=True, exist_ok=True)
        self._retro = None

    def _get_retro(self):
        if self._retro is None:
            # pyglet opens a GL shadow window at import time, which dies
            # without DISPLAY (the studio runs under systemd, headless) and
            # silently empties the built-in game list. Same guard as
            # sheeprl/_retro_run.py.
            import pyglet
            pyglet.options["shadow_window"] = False
            import retro
            self._retro = retro
        return self._retro

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_games(self) -> list[dict]:
        """List all available games: custom games first, then built-in retro games."""
        seen = set()
        games = []

        # 1. Custom games from games_dir (these take priority)
        for entry in sorted(self.games_dir.iterdir()):
            if not entry.is_dir():
                continue
            meta_path = entry / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
                game_id = meta.get("game_id") or entry.name
                meta["game_id"] = game_id
                meta["source"] = "custom"
                meta["has_custom_config"] = True
                meta["rom_ready"] = any(
                    p.suffix != ".sha" for p in entry.glob("rom.*")
                )
                seen.add(game_id)
                games.append(meta)
            except Exception as exc:
                print(f"[GameManager] Skipping {entry.name}: {exc}")

        # 2. Built-in stable-retro games
        try:
            retro = self._get_retro()
            builtin = retro.data.list_games(retro.data.Integrations.STABLE)
            for game_id in sorted(builtin):
                if game_id in seen:
                    continue
                system = _system_from_game_id(game_id)
                display = _display_name(game_id)

                # ROM presence: stable-retro ships integrations WITHOUT roms;
                # only hash-matched imports (python -m retro.import) fill them
                try:
                    retro.data.get_romfile_path(game_id)
                    rom_ready = True
                except Exception:
                    rom_ready = False

                games.append({
                    "game_id": game_id,
                    "display_name": display,
                    "system": system,
                    "source": "builtin",
                    "has_custom_config": False,
                    "rom_ready": rom_ready,
                })
        except Exception as exc:
            print(f"[GameManager] Could not list built-in games: {exc}")

        return games

    def get_game(self, game_id: str) -> dict:
        """Return full metadata + states for a game (custom or built-in)."""
        custom_dir = self.games_dir / game_id

        # Try custom first
        if (custom_dir / "metadata.json").exists():
            meta = json.loads((custom_dir / "metadata.json").read_text())
            meta["game_id"] = meta.get("game_id") or game_id
            meta["source"] = "custom"
            meta["has_custom_config"] = True
            # Annotated state list (label/group per save state) drives the
            # Watch tab; unannotated files on disk are merged in so nothing
            # is invisible. meta["states"] stays a plain filename list for
            # backwards compatibility.
            raw = meta.get("states")
            annotated = (
                raw if isinstance(raw, list) and raw and isinstance(raw[0], dict)
                else []
            )
            files = self.list_states(game_id)
            known = {s.get("file") for s in annotated}
            meta["annotated_states"] = annotated + [
                {"file": f, "label": f, "group": "other"}
                for f in files if f not in known
            ]
            meta["states"] = files
            meta["config_files"] = [
                fn for fn in VALID_CONFIG_FILENAMES
                if (custom_dir / fn).exists()
            ]
            return meta

        # Try built-in
        try:
            retro = self._get_retro()
            builtin_games = retro.data.list_games(retro.data.Integrations.STABLE)
            if game_id in builtin_games:
                system = _system_from_game_id(game_id)
                display = _display_name(game_id)

                states = retro.data.list_states(game_id, retro.data.Integrations.STABLE)
                # Read built-in data.json to show variables
                variables = []
                try:
                    data_path = retro.data.get_file_path(game_id, "data.json", retro.data.Integrations.STABLE)
                    if data_path:
                        with open(data_path) as f:
                            variables = list(json.load(f).get("info", {}).keys())
                except Exception:
                    pass

                return {
                    "game_id": game_id,
                    "display_name": display,
                    "system": system,
                    "source": "builtin",
                    "has_custom_config": False,
                    "default_state": states[0] if states else "",
                    "button_layout": _buttons_for_system(system),
                    "states": states,
                    "variables": variables,
                    "config_files": [],
                }
        except Exception:
            pass

        raise FileNotFoundError(f"Game '{game_id}' not found")

    def list_states(self, game_id: str) -> list[str]:
        """List states — custom states dir first, fall back to built-in."""
        # Custom states
        states_dir = self.games_dir / game_id / "states"
        if states_dir.exists():
            custom_states = sorted(p.stem for p in states_dir.glob("*.state"))
            if custom_states:
                return custom_states

        # Built-in states
        try:
            retro = self._get_retro()
            return retro.data.list_states(game_id, retro.data.Integrations.STABLE)
        except Exception:
            return []

    def game_dir(self, game_id: str) -> Path:
        return self.games_dir / game_id

    # ------------------------------------------------------------------
    # Config I/O
    # ------------------------------------------------------------------

    def read_config(self, game_id: str, filename: str) -> dict:
        """Read config — custom dir first, fall back to built-in retro data."""
        self._validate_filename(filename)

        # Check custom dir first
        custom_path = self.games_dir / game_id / filename
        if custom_path.exists():
            return json.loads(custom_path.read_text())

        # Fall back to built-in retro data for data.json, scenario.json, metadata.json
        if filename in ("data.json", "scenario.json", "metadata.json"):
            try:
                retro = self._get_retro()
                builtin_path = retro.data.get_file_path(
                    game_id, filename, retro.data.Integrations.STABLE
                )
                if builtin_path and os.path.exists(builtin_path):
                    return json.loads(Path(builtin_path).read_text())
            except Exception:
                pass

        # training.json and actions.json don't exist in built-in — return empty
        if filename == "training.json":
            return {"reward": {"variables": {}}, "done": {"variables": {}}}
        if filename == "actions.json":
            # Default: one no-op action (buttons are authored by NAME)
            return {"actions": [{"name": "NoOp", "buttons": []}]}

        raise FileNotFoundError(f"Config '{filename}' not found for game '{game_id}'")

    def write_config(self, game_id: str, filename: str, data: dict):
        """Write config — always writes to custom games dir (never modifies built-in).

        actions.json and training.json are validated HERE, at write time, so
        a broken config bounces back to the author (human or copilot) with a
        fix-it message instead of silently training a dead agent.
        """
        self._validate_filename(filename)
        if not isinstance(data, dict):
            raise ValueError(f"Config data must be a dict, got {type(data).__name__}")

        if filename == "actions.json":
            # Workspace metadata is authoritative for system (ids like
            # 'FZero-Test' don't encode one); fall back to id parsing.
            system = _system_from_game_id(game_id)
            meta_path = self.games_dir / game_id / "metadata.json"
            if meta_path.exists():
                try:
                    system = json.loads(meta_path.read_text()).get("system") or system
                except Exception:
                    pass
            buttons = _buttons_for_system(system)
            if not buttons:
                raise ValueError(
                    f"Can't validate actions for '{game_id}': unknown system "
                    f"'{system}' — game_id must end in -<System>-v<N> (e.g. -Nes-v0)"
                )
            _config_validation().resolve_action_mappings(
                data.get("actions", []), buttons, game_id
            )
        elif filename == "training.json":
            _config_validation().validate_training_config(game_id, data)

        game_dir = self.games_dir / game_id
        game_dir.mkdir(parents=True, exist_ok=True)

        path = game_dir / filename
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    # ------------------------------------------------------------------
    # Scaffolding
    # ------------------------------------------------------------------

    def scaffold_from_builtin(self, game_id: str) -> Path:
        """Create a custom games/ entry for a built-in game.

        Copies data.json, scenario.json, metadata.json from built-in,
        creates empty training.json and actions.json with system defaults.
        """
        game_dir = self.games_dir / game_id
        game_dir.mkdir(parents=True, exist_ok=True)
        (game_dir / "states").mkdir(exist_ok=True)

        retro = self._get_retro()
        system = _system_from_game_id(game_id)
        buttons = _buttons_for_system(system)

        # Copy built-in configs
        for fn in ("data.json", "scenario.json", "metadata.json"):
            if not (game_dir / fn).exists():
                try:
                    src = retro.data.get_file_path(game_id, fn, retro.data.Integrations.STABLE)
                    if src and os.path.exists(src):
                        import shutil
                        shutil.copy2(src, game_dir / fn)
                except Exception:
                    pass

        # Create metadata.json if not copied
        if not (game_dir / "metadata.json").exists():
            name_part = game_id.rsplit("-", 1)[0] if "-" in game_id else game_id
            display = ""
            for i, ch in enumerate(name_part):
                if ch.isupper() and i > 0 and name_part[i - 1].islower():
                    display += " "
                display += ch
            if system:
                display += f" ({system})"
            meta = {
                "display_name": display,
                "game_id": game_id,
                "system": system,
                "default_state": "",
                "button_layout": buttons,
            }
            (game_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

        # Create training.json (our custom reward shaping)
        if not (game_dir / "training.json").exists():
            (game_dir / "training.json").write_text(json.dumps(
                {"reward": {"variables": {}}, "done": {"variables": {}}}, indent=2
            ) + "\n")

        # Create actions.json with default per-button actions (authored by
        # NAME — index arrays are a retired format that let holes/mislabels
        # slip through)
        if not (game_dir / "actions.json").exists():
            actions = [{"name": "NoOp", "buttons": []}]
            for btn in buttons:
                if btn in ("Up", "Down", "Left", "Right", "B", "A"):
                    actions.append({"name": btn, "buttons": [btn]})
            (game_dir / "actions.json").write_text(json.dumps(
                {"actions": actions}, indent=2
            ) + "\n")

        print(f"[GameManager] Scaffolded custom config for '{game_id}' at {game_dir}")
        return game_dir

    def create_game(self, game_id: str, display_name: str, system: str) -> Path:
        """Scaffold a brand new game directory."""
        game_dir = self.games_dir / game_id
        if game_dir.exists():
            raise FileExistsError(f"Game directory already exists: {game_dir}")

        game_dir.mkdir(parents=True)
        (game_dir / "states").mkdir()

        buttons = _buttons_for_system(system)
        metadata = {
            "display_name": display_name,
            "game_id": game_id,
            "system": system,
            "default_state": "start",
            "button_layout": buttons,
        }
        (game_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        (game_dir / "data.json").write_text(json.dumps({"info": {}}, indent=2) + "\n")
        (game_dir / "scenario.json").write_text(json.dumps({
            "done": {"variables": {}}, "reward": {"variables": {}}
        }, indent=2) + "\n")
        (game_dir / "training.json").write_text(json.dumps({
            "reward": {"variables": {}}, "done": {"variables": {}}
        }, indent=2) + "\n")
        (game_dir / "actions.json").write_text(json.dumps({
            "actions": [{"name": "NoOp", "buttons": []}]
        }, indent=2) + "\n")

        print(f"[GameManager] Created game scaffold at {game_dir}")
        return game_dir

    def promote_game(self, game_id: str) -> dict:
        """Promote a ROM-ready built-in integration into a custom workspace.

        Copies the stock integration (RAM map, scenario, save states) plus the
        imported ROM into games/<id>/, then adds our scaffold files. After this
        the game is a first-class workspace: training.json rewards, reward_probe,
        state building, training — the same pipeline as any custom game."""
        import shutil

        game_dir = self.games_dir / game_id
        if game_dir.exists():
            raise FileExistsError(f"Workspace already exists: {game_dir}")

        retro = self._get_retro()
        try:
            rom_path = Path(retro.data.get_romfile_path(game_id))
        except Exception as exc:
            raise ValueError(
                f"{game_id} has no ROM installed in the built-in integration. "
                f"Import one first (python -m retro.import <folder>). ({exc})"
            )
        src = rom_path.parent

        system = _system_from_game_id(game_id)
        buttons = _buttons_for_system(system)

        game_dir.mkdir(parents=True)
        states_dir = game_dir / "states"
        states_dir.mkdir()

        copied, states = [], []
        for f in sorted(src.iterdir()):
            if f.suffix == ".state":
                shutil.copy2(f, states_dir / f.name)
                states.append(f.stem)
            elif f.name.startswith("rom.") or f.name in ("data.json", "scenario.json"):
                shutil.copy2(f, game_dir / f.name)
                copied.append(f.name)
            # skip script.lua / stock metadata.json — retro-specific, not ours

        metadata = {
            "display_name": _display_name(game_id),
            "game_id": game_id,
            "system": system,
            "default_state": states[0] if states else "start",
            "button_layout": buttons,
            "promoted_from": str(src),
        }
        (game_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (game_dir / "training.json").write_text(json.dumps({
            "reward": {"variables": {}}, "done": {"variables": {}}
        }, indent=2) + "\n")
        (game_dir / "actions.json").write_text(json.dumps({
            "actions": [{"name": "NoOp", "buttons": []}]
        }, indent=2) + "\n")

        ram_vars = []
        data_path = game_dir / "data.json"
        if data_path.exists():
            try:
                ram_vars = sorted(json.loads(data_path.read_text()).get("info", {}).keys())
            except Exception:
                pass

        print(f"[GameManager] Promoted built-in '{game_id}' to workspace {game_dir}")
        return {
            "game_dir": str(game_dir),
            "system": system,
            "copied": copied,
            "states": states,
            "ram_variables": ram_vars,
        }

    # ------------------------------------------------------------------
    # Retro integration
    # ------------------------------------------------------------------

    def setup_retro_integration(self, game_id: str):
        """Register custom integration path so retro can find our game configs."""
        retro = self._get_retro()
        custom_dir = self.games_dir / game_id
        if custom_dir.exists():
            retro.data.Integrations.add_custom_path(str(self.games_dir))
            print(f"[GameManager] Registered custom integration for '{game_id}'")

    def is_builtin(self, game_id: str) -> bool:
        """Check if a game exists in stable-retro's built-in library."""
        try:
            retro = self._get_retro()
            return game_id in retro.data.list_games(retro.data.Integrations.STABLE)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _game_dir(self, game_id: str) -> Path:
        return self.games_dir / game_id

    @staticmethod
    def _validate_filename(filename: str):
        if filename not in VALID_CONFIG_FILENAMES:
            raise ValueError(
                f"Invalid config filename '{filename}'. "
                f"Must be one of: {sorted(VALID_CONFIG_FILENAMES)}"
            )

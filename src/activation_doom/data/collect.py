from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from activation_doom.data.common import (
    DEFAULT_POLICY,
    dumps_json,
    should_accept,
    split_episodes,
    validate_records,
    visual_difference,
    write_diagnostics,
    write_json,
)
from activation_doom.preprocess import HEIGHT, WIDTH, save_target, target_gray


def button_name(button) -> str:
    """Return a stable ViZDoom button name."""
    return getattr(button, "name", str(button).split(".")[-1])


def var_name(var) -> str:
    """Return a stable ViZDoom game-variable name."""
    return getattr(var, "name", str(var).split(".")[-1])


def make_game(config: str, seed: int, game_variables=None, objects_info: bool = False):
    """Create and initialize one seeded ViZDoom game instance with optional structured state."""
    import vizdoom as vzd

    game = vzd.DoomGame()
    cfg = Path(config)
    if not cfg.exists():
        cfg = Path(vzd.scenarios_path) / config
    game.load_config(str(cfg))
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_seed(seed)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    if game_variables is not None:
        game.set_available_game_variables(game_variables)
    game.set_objects_info_enabled(objects_info)
    game.init()
    return game, str(cfg)


def action_vector(buttons: list[str], label: str, rng: random.Random) -> list[float]:
    """Build an action vector for a named stochastic exploration pattern."""
    values = [0.0] * len(buttons)

    def set_button(name: str, value: float = 1.0) -> None:
        """Set one available action button without failing on scenario omissions."""
        if name in buttons:
            values[buttons.index(name)] = value

    turn = rng.choice(["TURN_LEFT", "TURN_RIGHT"])
    lateral = rng.choice(["MOVE_LEFT", "MOVE_RIGHT"])
    if label == "forward":
        set_button("MOVE_FORWARD")
        set_button("SPEED", 1.0)
    elif label == "moving_turn":
        set_button("MOVE_FORWARD")
        set_button(turn)
    elif label == "lateral":
        set_button(lateral)
        if rng.random() < 0.5:
            set_button("MOVE_FORWARD")
    elif label == "combat":
        set_button("ATTACK")
        if rng.random() < 0.6:
            set_button("MOVE_FORWARD")
        if rng.random() < 0.5:
            set_button(turn)
    elif label == "turn":
        set_button(turn)
    elif label == "recovery":
        set_button("MOVE_BACKWARD")
        set_button(turn)
        set_button(lateral)
    else:
        for name in ["MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", "TURN_LEFT", "TURN_RIGHT", "ATTACK"]:
            if rng.random() < 0.25:
                set_button(name)
    return values


def choose_segment(policy: dict[str, float], rng: random.Random, recovery: bool) -> tuple[str, int]:
    """Choose an action label and segment length."""
    if recovery:
        return "recovery", rng.randint(8, 16)
    labels = list(policy)
    weights = [policy[k] for k in labels]
    return rng.choices(labels, weights=weights, k=1)[0], rng.randint(4, 18)


def parse_policy(text: str) -> dict[str, float]:
    """Parse comma-separated action policy weights."""
    if not text:
        return DEFAULT_POLICY
    policy = {}
    for part in text.split(","):
        key, value = part.split("=", 1)
        policy[key.strip()] = float(value)
    return policy


def state_vars(names: list[str], values) -> dict:
    """Convert ViZDoom game variable values into a JSON-serializable dictionary."""
    return {name.lower(): float(value) for name, value in zip(names, values)}


def important_event(previous: dict | None, current: dict) -> bool:
    """Detect cheap gameplay events worth saving even if visual novelty is low."""
    if previous is None:
        return False
    for key in ["health", "armor", "selected_weapon_ammo", "killcount"]:
        if key in current and current.get(key) != previous.get(key):
            return True
    return False


def move_to_split(root: Path, records: list[dict], split_by_episode: dict[int, str]) -> None:
    """Move staged frame files into train/val/test split directories and update metadata paths."""
    for split in ["train", "val", "test"]:
        (root / split / "original").mkdir(parents=True, exist_ok=True)
        (root / split / "processed").mkdir(parents=True, exist_ok=True)
    for record in records:
        split = split_by_episode[record["episode_id"]]
        record["dataset_split"] = split
        for key, folder in [("source_image_path", "original"), ("processed_image_path", "processed")]:
            old = root / record[key]
            new = root / split / folder / old.name
            shutil.move(str(old), str(new))
            record[key] = str(new.relative_to(root))
    shutil.rmtree(root / "_staging", ignore_errors=True)


def collect(args: argparse.Namespace) -> dict:
    """Collect a ViZDoom dataset and write metadata, diagnostics, and validation report."""
    root = Path(args.output)
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise SystemExit(f"{root} already exists; pass --overwrite to replace it")
    if root.exists() and args.overwrite:
        shutil.rmtree(root)
    (root / "_staging" / "original").mkdir(parents=True, exist_ok=True)
    (root / "_staging" / "processed").mkdir(parents=True, exist_ok=True)

    started = time.time()
    records: list[dict] = []
    candidate_frames = 0
    episode_id = 0
    episode_seeds: list[int] = []
    resolved_config = None
    policy = parse_policy(args.policy)
    episode_frame_cap = min(args.max_frames_per_episode, max(1, args.frames // 5))

    while len(records) < args.frames and episode_id < args.max_episodes:
        episode_seed = args.seed * 100000 + episode_id
        episode_seeds.append(episode_seed)
        game, resolved_config = make_game(args.scenario, episode_seed)
        buttons = [button_name(b) for b in game.get_available_buttons()]
        variables = [var_name(v) for v in game.get_available_game_variables()]
        policy_rng = random.Random(episode_seed)
        game.new_episode()

        last_saved = None
        last_tick = -10**9
        previous_tick_frame = None
        previous_vars = None
        low_change_ticks = 0
        segment_label, segment_left = choose_segment(policy, policy_rng, False)
        action = action_vector(buttons, segment_label, policy_rng)
        accepted_in_episode = 0
        episode_record_start = len(records)

        while not game.is_episode_finished() and len(records) < args.frames and accepted_in_episode < episode_frame_cap:
            if segment_left <= 0:
                recovery = low_change_ticks >= args.stuck_ticks
                segment_label, segment_left = choose_segment(policy, policy_rng, recovery)
                action = action_vector(buttons, segment_label, policy_rng)
                if recovery:
                    low_change_ticks = 0
            game.make_action(action, 1)
            segment_left -= 1
            if game.is_episode_finished():
                break
            state = game.get_state()
            if state is None:
                break

            candidate_frames += 1
            tick = int(game.get_episode_time())
            processed = target_gray(state.screen_buffer, args.width, args.height)
            tick_diff = visual_difference(previous_tick_frame, processed)
            previous_tick_frame = processed
            low_change_ticks = low_change_ticks + 1 if tick_diff is not None and tick_diff < args.stuck_threshold else 0

            vars_now = state_vars(variables, state.game_variables)
            reasons = ["episode_start"] if accepted_in_episode == 0 else []
            if important_event(previous_vars, vars_now):
                reasons.append("important_event")
            diff = visual_difference(last_saved, processed)
            reasons = should_accept(tick - last_tick, diff, args.minimum_save_interval, args.novelty_threshold, args.force_save_after, reasons)
            previous_vars = vars_now
            if not reasons:
                continue

            sample_id = f"e{episode_id:05d}_f{accepted_in_episode:05d}"
            source = root / "_staging" / "original" / f"{sample_id}.png"
            proc = root / "_staging" / "processed" / f"{sample_id}.png"
            Image.fromarray(state.screen_buffer, mode="RGB").save(source)
            save_target(proc, processed)
            records.append(
                {
                    "sample_id": sample_id,
                    "episode_id": episode_id,
                    "episode_seed": episode_seed,
                    "game_tick": tick,
                    "dataset_split": "pending",
                    "source_image_path": str(source.relative_to(root)),
                    "processed_image_path": str(proc.relative_to(root)),
                    "action_label": segment_label,
                    "action_vector": action,
                    "available_buttons": buttons,
                    "acceptance_reasons": reasons,
                    "visual_difference": diff,
                    "episode_finished": False,
                    "episode_truncated_by_frame_cap": False,
                    "player_dead": False,
                    **vars_now,
                }
            )
            last_saved = processed
            last_tick = tick
            accepted_in_episode += 1
        if len(records) > episode_record_start:
            records[-1]["episode_finished"] = bool(game.is_episode_finished())
            records[-1]["player_dead"] = bool(game.is_player_dead())
            records[-1]["episode_truncated_by_frame_cap"] = accepted_in_episode >= episode_frame_cap and not game.is_episode_finished()
        game.close()
        episode_id += 1

    split_map = split_episodes([r["episode_id"] for r in records], args.seed)
    move_to_split(root, records, split_map)
    metadata = "\n".join(dumps_json(r) for r in records) + "\n"
    (root / "metadata.jsonl").write_text(metadata, encoding="utf-8")

    config = {
        "scenario_config": resolved_config or args.scenario,
        "map": "scenario default",
        "dataset_seed": args.seed,
        "episode_seeds": episode_seeds,
        "target_sample_count": args.frames,
        "accepted_sample_count": len(records),
        "policy_probabilities": policy,
        "requested_max_frames_per_episode": args.max_frames_per_episode,
        "effective_max_frames_per_episode": episode_frame_cap,
        "minimum_save_interval": args.minimum_save_interval,
        "novelty_threshold": args.novelty_threshold,
        "force_save_after": args.force_save_after,
        "stuck_threshold": args.stuck_threshold,
        "stuck_ticks": args.stuck_ticks,
        "preprocessing": {"grayscale": True, "width": args.width, "height": args.height, "scale": "[0, 1]"},
        "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
    }
    write_json(root / "dataset_config.json", config)
    report = write_diagnostics(root, records, config, started, time.time(), candidate_frames)
    validation = validate_records(root, records, args.frames if len(records) == args.frames else None)
    write_json(root / "validation_report.json", validation)
    print(json.dumps({"output": str(root), "report": report, "validation": validation}, indent=2, sort_keys=True))
    return {"records": records, "report": report, "validation": validation}


def parser() -> argparse.ArgumentParser:
    """Build the dataset collection CLI parser."""
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--frames", type=int, default=25000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scenario", default="deathmatch.cfg")
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--height", type=int, default=HEIGHT)
    p.add_argument("--minimum-save-interval", type=int, default=3)
    p.add_argument("--novelty-threshold", type=float, default=0.015)
    p.add_argument("--force-save-after", type=int, default=12)
    p.add_argument("--stuck-threshold", type=float, default=0.003)
    p.add_argument("--stuck-ticks", type=int, default=30)
    p.add_argument("--policy", default="")
    p.add_argument("--max-frames-per-episode", type=int, default=500)
    p.add_argument("--max-episodes", type=int, default=1000)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    """Run dataset collection from command-line arguments."""
    collect(parser().parse_args())


if __name__ == "__main__":
    main()

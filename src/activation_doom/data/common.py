from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from activation_doom.preprocess import HEIGHT, WIDTH, load_target, target_gray


DEFAULT_POLICY = {
    "forward": 0.35,
    "moving_turn": 0.20,
    "lateral": 0.15,
    "combat": 0.15,
    "turn": 0.10,
    "random": 0.05,
}


def dumps_json(data: dict) -> str:
    """Serialize a dictionary as one compact JSON line."""
    return json.dumps(data, sort_keys=True)


def read_jsonl(path: Path) -> list[dict]:
    """Read metadata records from a JSONL file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: dict) -> None:
    """Write a dictionary as pretty JSON."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def visual_difference(a: np.ndarray | None, b: np.ndarray) -> float | None:
    """Return mean absolute pixel difference for normalized grayscale frames."""
    if a is None:
        return None
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def should_accept(ticks_since_save: int, diff: float | None, min_interval: int, threshold: float, force_after: int, reasons: list[str]) -> list[str]:
    """Return acceptance reasons for a candidate frame or an empty list."""
    if "episode_start" in reasons:
        return reasons
    if "important_event" in reasons:
        return reasons
    if ticks_since_save >= force_after:
        return reasons + ["forced_interval"]
    if diff is not None and ticks_since_save >= min_interval and diff >= threshold:
        return reasons + ["novelty"]
    return []


def split_episodes(episode_ids: list[int], seed: int, ratios=(0.8, 0.1, 0.1)) -> dict[int, str]:
    """Assign whole episodes to train, val, or test splits deterministically."""
    ids = sorted(set(episode_ids))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    if n >= 3:
        n_val = max(1, int(n * ratios[1]))
        n_test = max(1, int(n * ratios[2]))
        n_train = max(1, n - n_val - n_test)
    else:
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
    train = set(ids[:n_train])
    val = set(ids[n_train : n_train + n_val])
    return {episode_id: ("train" if episode_id in train else "val" if episode_id in val else "test") for episode_id in ids}


def disk_usage(path: Path) -> int:
    """Return total file bytes below a directory."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def make_contact_sheet(paths: list[Path], out: Path, cols: int = 10) -> None:
    """Create a tiled contact sheet from processed grayscale image paths."""
    if not paths:
        Image.new("L", (WIDTH, HEIGHT), 0).save(out)
        return
    rows = int(np.ceil(len(paths) / cols))
    sheet = Image.new("L", (cols * WIDTH, rows * HEIGHT), 0)
    for i, path in enumerate(paths):
        sheet.paste(Image.open(path).convert("L"), ((i % cols) * WIDTH, (i // cols) * HEIGHT))
    sheet.save(out)


def plot_bar(counter: Counter, out: Path, title: str) -> None:
    """Save a simple bar chart for counted categories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(counter.keys())
    values = [counter[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.8), 3), dpi=150)
    ax.bar(labels, values)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_hist(values: list[float], out: Path, title: str) -> None:
    """Save a histogram for numeric diagnostic values."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    ax.hist(values, bins=40)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_diagnostics(root: Path, records: list[dict], config: dict, started: float, finished: float, candidate_frames: int) -> dict:
    """Write reports and plots summarizing a collected dataset."""
    diag = root / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config["dataset_seed"])
    paths = [root / r["processed_image_path"] for r in records]
    make_contact_sheet([*rng.sample(paths, min(100, len(paths)))], diag / "contact_sheet.png")
    ordered = []
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_episode[record["episode_id"]].append(record)
    for episode_id in sorted(by_episode)[:5]:
        episode = by_episode[episode_id]
        step = max(1, len(episode) // 20)
        ordered.extend(root / r["processed_image_path"] for r in episode[::step][:20])
    make_contact_sheet(ordered[:100], diag / "episode_contact_sheet.png")

    diffs = [r["visual_difference"] for r in records if r["visual_difference"] is not None]
    actions = Counter(r["action_label"] for r in records)
    reasons = Counter(reason for r in records for reason in r["acceptance_reasons"])
    episode_counts = Counter(r["episode_id"] for r in records)
    plot_hist(diffs, diag / "frame_difference_distribution.png", "Accepted-frame visual differences")
    plot_bar(actions, diag / "action_distribution.png", "Action distribution")
    plot_bar(Counter({str(k): v for k, v in episode_counts.items()}), diag / "episode_distribution.png", "Frames per episode")

    splits = Counter(r["dataset_split"] for r in records)
    variable_stats = {}
    for key in ["health", "armor", "selected_weapon_ammo", "killcount"]:
        values = [r[key] for r in records if key in r]
        if values:
            variable_stats[key] = {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
    report = {
        "total_accepted_frames": len(records),
        "total_candidate_frames": candidate_frames,
        "rejected_candidate_frames": max(0, candidate_frames - len(records)),
        "ended_early": len(records) < config["target_sample_count"],
        "early_stop_reason": None if len(records) >= config["target_sample_count"] else "max_episodes_or_episode_timeouts",
        "episodes": len(episode_counts),
        "frames_per_split": dict(splits),
        "frames_per_episode_min": min(episode_counts.values()) if episode_counts else 0,
        "frames_per_episode_max": max(episode_counts.values()) if episode_counts else 0,
        "frames_per_episode_mean": float(np.mean(list(episode_counts.values()))) if episode_counts else 0.0,
        "action_frequencies": dict(actions),
        "acceptance_reason_frequencies": dict(reasons),
        "visual_difference_mean": float(np.mean(diffs)) if diffs else None,
        "visual_difference_median": float(np.median(diffs)) if diffs else None,
        "nearly_static_percentage": float(np.mean([d < config["novelty_threshold"] for d in diffs]) * 100.0) if diffs else None,
        "game_variable_stats": variable_stats,
        "episode_finished_count": sum(1 for r in records if r.get("episode_finished")),
        "player_death_count": sum(1 for r in records if r.get("player_dead")),
        "episode_truncated_by_frame_cap_count": sum(1 for r in records if r.get("episode_truncated_by_frame_cap")),
        "collection_seconds": finished - started,
        "accepted_frames_per_second": len(records) / max(1e-6, finished - started),
        "disk_usage_bytes": disk_usage(root),
    }
    write_json(diag / "dataset_report.json", report)
    (diag / "dataset_report.md").write_text(
        "\n".join(
            [
                "# Dataset Report",
                "",
                f"- Accepted frames: {report['total_accepted_frames']}",
                f"- Candidate frames: {report['total_candidate_frames']}",
                f"- Episodes: {report['episodes']}",
                f"- Splits: {report['frames_per_split']}",
                f"- Mean visual diff: {report['visual_difference_mean']}",
                f"- Median visual diff: {report['visual_difference_median']}",
                f"- Disk bytes: {report['disk_usage_bytes']}",
            ]
        ),
        encoding="utf-8",
    )
    return report


def validate_records(root: Path, records: list[dict], expected_frames: int | None = None) -> dict:
    """Validate dataset metadata, split isolation, image paths, and preprocessing determinism."""
    errors = []
    sample_ids = [r["sample_id"] for r in records]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("duplicate sample IDs")
    split_by_episode: dict[int, set[str]] = defaultdict(set)
    duplicate_hashes = Counter()
    for record in records:
        split_by_episode[record["episode_id"]].add(record["dataset_split"])
        original = root / record["source_image_path"]
        processed = root / record["processed_image_path"]
        if not original.exists() or not processed.exists():
            errors.append(f"missing image path for {record['sample_id']}")
            continue
        img = Image.open(processed)
        if img.mode != "L" or img.size != (WIDTH, HEIGHT):
            errors.append(f"bad processed image shape/mode for {record['sample_id']}")
        arr = load_target(processed)
        if arr.min() < 0.0 or arr.max() > 1.0:
            errors.append(f"bad processed range for {record['sample_id']}")
        regenerated = target_gray(Image.open(original))
        if not np.array_equal(np.rint(regenerated * 255).astype(np.uint8), np.asarray(img)):
            errors.append(f"non-deterministic preprocessing for {record['sample_id']}")
        duplicate_hashes[img.tobytes()] += 1
    leaked = [episode for episode, splits in split_by_episode.items() if len(splits) > 1]
    if leaked:
        errors.append(f"episodes in multiple splits: {leaked[:5]}")
    if expected_frames is not None and len(records) != expected_frames:
        errors.append(f"expected {expected_frames} frames, found {len(records)}")
    exact_duplicates = sum(count - 1 for count in duplicate_hashes.values() if count > 1)
    return {"ok": not errors, "errors": errors, "records": len(records), "exact_duplicate_processed_frames": exact_duplicates}

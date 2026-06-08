#!/usr/bin/env python
"""Frame-level string/fret diagnostics for SynthTab checkpoints.

This is an evaluation-only tool. It loads a trained checkpoint and a small
validation subset, runs inference, and measures how often the model gets the
pitch right but places it on the wrong string/fret.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import amt_tools.tools as tools
from amt_tools.inference import run_offline


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from tabcnn_synthtab_pipeline import (  # noqa: E402
    build_estimators,
    build_feature_extractor,
    build_model,
    build_profile,
    create_synthtab_dataset,
    load_config,
    resolve_device,
    safe_torch_load,
)


METRIC_LINE = re.compile(r"^\s*([A-Za-z0-9_-]+)\s+:\s+([-+0-9.eE]+)\s*$")


@dataclass(frozen=True)
class TrackCandidate:
    track_id: str
    bucket: str
    multi_pitch_f1: float | None
    tablature_f1: float | None
    gap: float | None
    ref_silence_ratio: float | None = None


@dataclass
class Counts:
    ref_notes: int = 0
    pred_notes: int = 0
    exact_tp: int = 0
    pitch_tp: int = 0
    canonical_tp: int = 0
    active_frames_ref: int = 0
    active_frames_pred: int = 0
    frames: int = 0

    def add(self, other: "Counts") -> None:
        self.ref_notes += other.ref_notes
        self.pred_notes += other.pred_notes
        self.exact_tp += other.exact_tp
        self.pitch_tp += other.pitch_tp
        self.canonical_tp += other.canonical_tp
        self.active_frames_ref += other.active_frames_ref
        self.active_frames_pred += other.active_frames_pred
        self.frames += other.frames

    @property
    def pitch_correct_wrong_position(self) -> int:
        return max(0, self.pitch_tp - self.exact_tp)

    @property
    def missed_notes(self) -> int:
        return max(0, self.ref_notes - self.pitch_tp)

    @property
    def extra_notes(self) -> int:
        return max(0, self.pred_notes - self.pitch_tp)


@dataclass
class TopKCounts:
    ref_notes: int = 0
    pitch_correct_wrong_notes: int = 0
    exact_matched_notes: int = 0
    extra_notes: int = 0
    correct_top1: int = 0
    correct_top2: int = 0
    correct_top3: int = 0
    correct_top5: int = 0
    pitch_compatible_top1: int = 0
    pitch_compatible_top2: int = 0
    pitch_compatible_top3: int = 0
    pitch_compatible_top5: int = 0
    pcw_correct_top1: int = 0
    pcw_correct_top2: int = 0
    pcw_correct_top3: int = 0
    pcw_correct_top5: int = 0
    pcw_pitch_compatible_top1: int = 0
    pcw_pitch_compatible_top2: int = 0
    pcw_pitch_compatible_top3: int = 0
    pcw_pitch_compatible_top5: int = 0
    correct_rank_sum: float = 0.0
    pitch_compatible_rank_sum: float = 0.0
    pcw_correct_rank_sum: float = 0.0
    pcw_pitch_compatible_rank_sum: float = 0.0
    correct_prob_sum: float = 0.0
    string_top1_prob_sum: float = 0.0
    pcw_margin_top1_minus_correct_sum: float = 0.0
    extra_note_margin_sum: float = 0.0
    extra_note_prob_sum: float = 0.0
    extra_silence_prob_sum: float = 0.0
    weak_extra_margin_005: int = 0
    weak_extra_margin_010: int = 0

    def add(self, other: "TopKCounts") -> None:
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))


@dataclass
class TrackReport:
    track_id: str
    bucket: str
    source_multi_pitch_f1: float | None
    source_tablature_f1: float | None
    source_gap: float | None
    source_ref_silence_ratio: float | None
    counts: Counts
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)


@dataclass
class TopKTrackReport:
    track_id: str
    bucket: str
    source_multi_pitch_f1: float | None
    source_tablature_f1: float | None
    source_gap: float | None
    source_ref_silence_ratio: float | None
    counts: TopKCounts
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodeVariant:
    name: str
    top_k: int
    transition_penalty: float
    duplicate_pitch_penalty: float = 0.0


DECODE_VARIANTS = [
    DecodeVariant("baseline_top1", top_k=1, transition_penalty=0.0),
    DecodeVariant("topk3_smooth_light", top_k=3, transition_penalty=0.05),
    DecodeVariant("topk5_smooth_light", top_k=5, transition_penalty=0.05),
    DecodeVariant("topk5_smooth_medium", top_k=5, transition_penalty=0.15),
    DecodeVariant("topk5_smooth_strong", top_k=5, transition_penalty=0.35),
    DecodeVariant("topk5_smooth_medium_dup", top_k=5, transition_penalty=0.15, duplicate_pitch_penalty=0.20),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_metric_txt(path: Path) -> dict[str, dict[str, float]]:
    sections: dict[str, dict[str, float]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("-----") and stripped.endswith("-----"):
            current = stripped.strip("-").strip()
            sections.setdefault(current, {})
            continue
        if current is None:
            continue
        match = METRIC_LINE.match(line)
        if match:
            key, raw = match.groups()
            sections[current][key] = float(raw)
    return sections


def metric_value(section: dict[str, float], key: str) -> float | None:
    value = section.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def f1(tp: int, pred: int, ref: int) -> float:
    precision = rate(tp, pred)
    recall = rate(tp, ref)
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def summarize_counts(counts: Counts) -> dict[str, float | int]:
    return {
        "frames": counts.frames,
        "ref_notes": counts.ref_notes,
        "pred_notes": counts.pred_notes,
        "exact_tp": counts.exact_tp,
        "pitch_tp": counts.pitch_tp,
        "pitch_correct_wrong_position": counts.pitch_correct_wrong_position,
        "missed_notes": counts.missed_notes,
        "extra_notes": counts.extra_notes,
        "exact_precision": rate(counts.exact_tp, counts.pred_notes),
        "exact_recall": rate(counts.exact_tp, counts.ref_notes),
        "exact_f1": f1(counts.exact_tp, counts.pred_notes, counts.ref_notes),
        "pitch_precision": rate(counts.pitch_tp, counts.pred_notes),
        "pitch_recall": rate(counts.pitch_tp, counts.ref_notes),
        "pitch_f1": f1(counts.pitch_tp, counts.pred_notes, counts.ref_notes),
        "oracle_tab_f1": f1(counts.pitch_tp, counts.pred_notes, counts.ref_notes),
        "canonical_f1": f1(counts.canonical_tp, counts.pred_notes, counts.ref_notes),
        "wrong_position_share_of_ref": rate(counts.pitch_correct_wrong_position, counts.ref_notes),
        "missed_share_of_ref": rate(counts.missed_notes, counts.ref_notes),
        "extra_share_of_pred": rate(counts.extra_notes, counts.pred_notes),
        "ref_active_frame_ratio": rate(counts.active_frames_ref, counts.frames),
        "pred_active_frame_ratio": rate(counts.active_frames_pred, counts.frames),
    }


def summarize_topk_counts(counts: TopKCounts) -> dict[str, float | int]:
    return {
        "ref_notes": counts.ref_notes,
        "pitch_correct_wrong_notes": counts.pitch_correct_wrong_notes,
        "exact_matched_notes": counts.exact_matched_notes,
        "extra_notes": counts.extra_notes,
        "correct_top1_rate": rate(counts.correct_top1, counts.ref_notes),
        "correct_top2_rate": rate(counts.correct_top2, counts.ref_notes),
        "correct_top3_rate": rate(counts.correct_top3, counts.ref_notes),
        "correct_top5_rate": rate(counts.correct_top5, counts.ref_notes),
        "pitch_compatible_top1_rate": rate(counts.pitch_compatible_top1, counts.ref_notes),
        "pitch_compatible_top2_rate": rate(counts.pitch_compatible_top2, counts.ref_notes),
        "pitch_compatible_top3_rate": rate(counts.pitch_compatible_top3, counts.ref_notes),
        "pitch_compatible_top5_rate": rate(counts.pitch_compatible_top5, counts.ref_notes),
        "pcw_correct_top1_rate": rate(counts.pcw_correct_top1, counts.pitch_correct_wrong_notes),
        "pcw_correct_top2_rate": rate(counts.pcw_correct_top2, counts.pitch_correct_wrong_notes),
        "pcw_correct_top3_rate": rate(counts.pcw_correct_top3, counts.pitch_correct_wrong_notes),
        "pcw_correct_top5_rate": rate(counts.pcw_correct_top5, counts.pitch_correct_wrong_notes),
        "pcw_pitch_compatible_top1_rate": rate(counts.pcw_pitch_compatible_top1, counts.pitch_correct_wrong_notes),
        "pcw_pitch_compatible_top2_rate": rate(counts.pcw_pitch_compatible_top2, counts.pitch_correct_wrong_notes),
        "pcw_pitch_compatible_top3_rate": rate(counts.pcw_pitch_compatible_top3, counts.pitch_correct_wrong_notes),
        "pcw_pitch_compatible_top5_rate": rate(counts.pcw_pitch_compatible_top5, counts.pitch_correct_wrong_notes),
        "avg_correct_rank": rate(counts.correct_rank_sum, counts.ref_notes),
        "avg_pitch_compatible_rank": rate(counts.pitch_compatible_rank_sum, counts.ref_notes),
        "avg_pcw_correct_rank": rate(counts.pcw_correct_rank_sum, counts.pitch_correct_wrong_notes),
        "avg_pcw_pitch_compatible_rank": rate(counts.pcw_pitch_compatible_rank_sum, counts.pitch_correct_wrong_notes),
        "avg_correct_prob": rate(counts.correct_prob_sum, counts.ref_notes),
        "avg_string_top1_prob": rate(counts.string_top1_prob_sum, counts.ref_notes),
        "avg_pcw_top1_minus_correct_prob": rate(
            counts.pcw_margin_top1_minus_correct_sum,
            counts.pitch_correct_wrong_notes,
        ),
        "avg_extra_note_margin": rate(counts.extra_note_margin_sum, counts.extra_notes),
        "avg_extra_note_prob": rate(counts.extra_note_prob_sum, counts.extra_notes),
        "avg_extra_silence_prob": rate(counts.extra_silence_prob_sum, counts.extra_notes),
        "weak_extra_margin_005_rate": rate(counts.weak_extra_margin_005, counts.extra_notes),
        "weak_extra_margin_010_rate": rate(counts.weak_extra_margin_010, counts.extra_notes),
    }


def experiment_results_dir(experiment_dir: Path) -> Path:
    results_dir = experiment_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")
    return results_dir


def load_track_candidates(experiment_dir: Path, max_tracks: int, max_ref_silence: float) -> list[TrackCandidate]:
    results_dir = experiment_results_dir(experiment_dir)
    summary = load_json(results_dir / "summary.json")
    diagnostics = summary.get("synthtab_val", {}).get("diagnostics", {})
    diagnostic_tracks = {
        track.get("track_id"): track
        for track in diagnostics.get("tracks", [])
        if isinstance(track, dict) and isinstance(track.get("track_id"), str)
    }

    candidates: list[TrackCandidate] = []
    for txt_path in sorted(results_dir.rglob("*.txt")):
        sections = parse_metric_txt(txt_path)
        mp = metric_value(sections.get("multi_pitch", {}), "f1-score")
        tab = metric_value(sections.get("tablature", {}), "f1-score")
        gap = mp - tab if mp is not None and tab is not None else None
        track_id = str(txt_path.relative_to(results_dir).with_suffix(""))
        ref_silence = metric_value(diagnostic_tracks.get(track_id, {}), "ref_silence_ratio")
        candidates.append(TrackCandidate(track_id, "", mp, tab, gap, ref_silence))

    ranked = [
        candidate
        for candidate in candidates
        if candidate.gap is not None
        and candidate.tablature_f1 is not None
        and (candidate.ref_silence_ratio is None or candidate.ref_silence_ratio <= max_ref_silence)
    ]
    if not ranked:
        raise ValueError(f"No per-track metrics were found under {results_dir}")

    worst = sorted(ranked, key=lambda candidate: candidate.gap or 0.0, reverse=True)[:10]
    best = sorted(ranked, key=lambda candidate: candidate.tablature_f1 or 0.0, reverse=True)[:5]
    median_tab = float(np.median([candidate.tablature_f1 for candidate in ranked if candidate.tablature_f1 is not None]))
    typical = sorted(ranked, key=lambda candidate: abs((candidate.tablature_f1 or 0.0) - median_tab))[:5]

    selected: list[TrackCandidate] = []
    seen: set[str] = set()
    for bucket, group in [("worst_gap", worst), ("typical_tab", typical), ("best_tab", best)]:
        for candidate in group:
            if candidate.track_id in seen:
                continue
            selected.append(
                TrackCandidate(
                    candidate.track_id,
                    bucket,
                    candidate.multi_pitch_f1,
                    candidate.tablature_f1,
                    candidate.gap,
                    candidate.ref_silence_ratio,
                )
            )
            seen.add(candidate.track_id)
            if len(selected) >= max_tracks:
                return selected
    return selected


def canonical_positions_for_pitches(profile: tools.GuitarProfile, pitch_counts: Counter[int]) -> Counter[tuple[int, int]]:
    positions: Counter[tuple[int, int]] = Counter()
    used: set[tuple[int, int]] = set()
    for pitch, count in pitch_counts.items():
        valid_positions = sorted(profile.get_valid_positions(pitch), key=lambda item: (item[1], item[0]))
        for _ in range(count):
            chosen = None
            for position in valid_positions:
                if position not in used:
                    chosen = position
                    break
            if chosen is None and valid_positions:
                chosen = valid_positions[0]
            if chosen is not None:
                positions[chosen] += 1
                used.add(chosen)
    return positions


def frame_positions(tab: np.ndarray, frame_idx: int) -> Counter[tuple[int, int]]:
    positions: Counter[tuple[int, int]] = Counter()
    for string_idx in range(tab.shape[0]):
        fret = int(tab[string_idx, frame_idx])
        if fret >= 0:
            positions[(string_idx, fret)] += 1
    return positions


def pitch_counts(profile: tools.GuitarProfile, positions: Counter[tuple[int, int]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for (string_idx, fret), count in positions.items():
        counts[int(profile.get_pitch(string_idx, fret))] += count
    return counts


def counter_intersection_count(left: Counter[Any], right: Counter[Any]) -> int:
    return int(sum((left & right).values()))


def normalize_tab(tab: Any) -> np.ndarray:
    arr = np.asarray(tab)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected tablature with 2 dims, got shape {arr.shape}")
    if arr.shape[0] <= arr.shape[1]:
        return arr
    return arr.T


def compare_tablature(reference: np.ndarray, estimated: np.ndarray, profile: tools.GuitarProfile) -> Counts:
    ref = normalize_tab(reference)
    pred = normalize_tab(estimated)
    frames = min(ref.shape[-1], pred.shape[-1])
    counts = Counts(frames=frames)

    for frame_idx in range(frames):
        ref_positions = frame_positions(ref, frame_idx)
        pred_positions = frame_positions(pred, frame_idx)
        ref_pitches = pitch_counts(profile, ref_positions)
        pred_pitches = pitch_counts(profile, pred_positions)
        canonical_positions = canonical_positions_for_pitches(profile, pred_pitches)

        counts.ref_notes += int(sum(ref_positions.values()))
        counts.pred_notes += int(sum(pred_positions.values()))
        counts.exact_tp += counter_intersection_count(ref_positions, pred_positions)
        counts.pitch_tp += counter_intersection_count(ref_pitches, pred_pitches)
        counts.canonical_tp += counter_intersection_count(ref_positions, canonical_positions)
        counts.active_frames_ref += int(bool(ref_positions))
        counts.active_frames_pred += int(bool(pred_positions))

    return counts


def rank_descending(scores: np.ndarray, target_index: int) -> int:
    target_score = float(scores[target_index])
    return int(np.sum(scores > target_score) + 1)


def increment_topk(counts: TopKCounts, prefix: str, rank: int) -> None:
    for k in [1, 2, 3, 5]:
        if rank <= k:
            field_name = f"{prefix}_top{k}"
            setattr(counts, field_name, getattr(counts, field_name) + 1)


def best_pitch_compatible_rank(
    frame_probs: np.ndarray,
    profile: tools.GuitarProfile,
    pitch: int,
    silence_class: int,
) -> int | None:
    valid_positions = [
        (string_idx, fret)
        for string_idx, fret in profile.get_valid_positions(pitch)
        if 0 <= fret < silence_class
    ]
    if not valid_positions:
        return None

    note_scores = frame_probs[:, :silence_class].reshape(-1)
    best_rank = None
    for string_idx, fret in valid_positions:
        flat_index = string_idx * silence_class + fret
        rank = rank_descending(note_scores, flat_index)
        if best_rank is None or rank < best_rank:
            best_rank = rank
    return best_rank


def run_raw_tablature(track_data: dict[str, Any], model: Any) -> tuple[np.ndarray, np.ndarray]:
    track_data = tools.dict_to_dtype(track_data, dtype=tools.FLOAT32)
    batch = tools.dict_unsqueeze(tools.dict_to_tensor(track_data))

    with torch.no_grad():
        batch = model.pre_proc(batch)
        raw_logits = model(batch[tools.KEY_FEATS])[tools.KEY_TABLATURE]
        output_layer = model.dense[-1]
        predicted = output_layer.finalize_output(raw_logits)
        logits_grouped = raw_logits.view(
            raw_logits.size(0),
            raw_logits.size(1),
            output_layer.num_groups,
            output_layer.num_classes,
        )
        probabilities = torch.softmax(logits_grouped, dim=-1)

    return (
        predicted.detach().cpu().numpy()[0],
        probabilities.detach().cpu().numpy()[0],
    )


def analyze_topk(
    reference: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    profile: tools.GuitarProfile,
) -> TopKCounts:
    ref = normalize_tab(reference)
    pred = normalize_tab(predicted)
    probs = np.asarray(probabilities)
    if probs.ndim != 3:
        raise ValueError(f"Expected probabilities with shape (T, S, C), got {probs.shape}")

    frames = min(ref.shape[-1], pred.shape[-1], probs.shape[0])
    silence_class = probs.shape[-1] - 1
    counts = TopKCounts()

    for frame_idx in range(frames):
        ref_positions = frame_positions(ref, frame_idx)
        pred_positions = frame_positions(pred, frame_idx)
        pred_pitch_matches = pitch_counts(profile, pred_positions)

        for (string_idx, fret), ref_count in ref_positions.items():
            pitch = int(profile.get_pitch(string_idx, fret))
            string_scores = probs[frame_idx, string_idx]
            correct_rank = rank_descending(string_scores, fret)
            pitch_rank = best_pitch_compatible_rank(probs[frame_idx], profile, pitch, silence_class)
            top1_prob = float(np.max(string_scores))
            correct_prob = float(string_scores[fret])

            for _ in range(ref_count):
                counts.ref_notes += 1
                counts.correct_rank_sum += correct_rank
                counts.correct_prob_sum += correct_prob
                counts.string_top1_prob_sum += top1_prob
                increment_topk(counts, "correct", correct_rank)
                if pitch_rank is not None:
                    counts.pitch_compatible_rank_sum += pitch_rank
                    increment_topk(counts, "pitch_compatible", pitch_rank)

                exact_available = pred_positions.get((string_idx, fret), 0) > 0
                pitch_available = pred_pitch_matches.get(pitch, 0) > 0
                if exact_available:
                    counts.exact_matched_notes += 1
                    pred_positions[(string_idx, fret)] -= 1
                    pred_pitch_matches[pitch] -= 1
                elif pitch_available:
                    counts.pitch_correct_wrong_notes += 1
                    pred_pitch_matches[pitch] -= 1
                    counts.pcw_correct_rank_sum += correct_rank
                    counts.pcw_margin_top1_minus_correct_sum += top1_prob - correct_prob
                    increment_topk(counts, "pcw_correct", correct_rank)
                    if pitch_rank is not None:
                        counts.pcw_pitch_compatible_rank_sum += pitch_rank
                        increment_topk(counts, "pcw_pitch_compatible", pitch_rank)

        ref_pitch_matches = pitch_counts(profile, ref_positions)
        for (string_idx, fret), pred_count in pred_positions.items():
            if pred_count <= 0:
                continue
            pitch = int(profile.get_pitch(string_idx, fret))
            for _ in range(pred_count):
                if ref_pitch_matches.get(pitch, 0) > 0:
                    ref_pitch_matches[pitch] -= 1
                    continue
                note_prob = float(probs[frame_idx, string_idx, fret])
                silence_prob = float(probs[frame_idx, string_idx, silence_class])
                margin = note_prob - silence_prob
                counts.extra_notes += 1
                counts.extra_note_prob_sum += note_prob
                counts.extra_silence_prob_sum += silence_prob
                counts.extra_note_margin_sum += margin
                counts.weak_extra_margin_005 += int(margin <= 0.05)
                counts.weak_extra_margin_010 += int(margin <= 0.10)

    return counts


def decode_baseline_top1(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities)
    if probs.ndim != 3:
        raise ValueError(f"Expected probabilities with shape (T, S, C), got {probs.shape}")
    silence_class = probs.shape[-1] - 1
    decoded = np.argmax(probs, axis=-1).T.astype(np.int64)
    decoded[decoded == silence_class] = -1
    return decoded


def topk_candidates_for_string(
    probs: np.ndarray,
    top_k: int,
    silence_class: int,
) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for frame_idx in range(probs.shape[0]):
        frame_probs = probs[frame_idx]
        top = np.argsort(frame_probs)[::-1][:top_k]
        if silence_class not in top:
            top = np.append(top, silence_class)
        candidates.append(np.unique(top))
    return candidates


def transition_cost(prev_class: int, cur_class: int, silence_class: int, penalty: float) -> float:
    if prev_class == cur_class or penalty <= 0:
        return 0.0
    if prev_class == silence_class or cur_class == silence_class:
        return penalty
    return penalty * min(4.0, abs(cur_class - prev_class))


def viterbi_decode_string(probs: np.ndarray, top_k: int, transition_penalty: float, silence_class: int) -> np.ndarray:
    eps = 1e-12
    candidates = topk_candidates_for_string(probs, top_k=top_k, silence_class=silence_class)
    scores: dict[int, float] = {int(cls): float(np.log(probs[0, cls] + eps)) for cls in candidates[0]}
    backpointers: list[dict[int, int]] = []

    for frame_idx in range(1, probs.shape[0]):
        current_scores: dict[int, float] = {}
        current_backpointers: dict[int, int] = {}
        for cur in candidates[frame_idx]:
            cur_int = int(cur)
            emit = float(np.log(probs[frame_idx, cur_int] + eps))
            best_prev = None
            best_score = -float("inf")
            for prev, prev_score in scores.items():
                score = prev_score + emit - transition_cost(prev, cur_int, silence_class, transition_penalty)
                if score > best_score:
                    best_score = score
                    best_prev = prev
            current_scores[cur_int] = best_score
            current_backpointers[cur_int] = int(best_prev if best_prev is not None else silence_class)
        scores = current_scores
        backpointers.append(current_backpointers)

    last = max(scores, key=scores.get)
    path = [last]
    for frame_idx in range(len(backpointers) - 1, -1, -1):
        last = backpointers[frame_idx][last]
        path.append(last)
    path.reverse()
    return np.asarray(path, dtype=np.int64)


def suppress_duplicate_pitches(
    decoded_classes: np.ndarray,
    probabilities: np.ndarray,
    profile: tools.GuitarProfile,
    penalty: float,
) -> np.ndarray:
    if penalty <= 0:
        return decoded_classes
    decoded = decoded_classes.copy()
    silence_class = probabilities.shape[-1] - 1
    frames, strings = decoded.shape

    for frame_idx in range(frames):
        by_pitch: dict[int, list[tuple[int, int, float]]] = {}
        for string_idx in range(strings):
            fret = int(decoded[frame_idx, string_idx])
            if fret == silence_class:
                continue
            pitch = int(profile.get_pitch(string_idx, fret))
            prob = float(probabilities[frame_idx, string_idx, fret])
            by_pitch.setdefault(pitch, []).append((string_idx, fret, prob))

        for duplicates in by_pitch.values():
            if len(duplicates) <= 1:
                continue
            keep = max(duplicates, key=lambda item: item[2])
            for string_idx, fret, prob in duplicates:
                if (string_idx, fret, prob) == keep:
                    continue
                silence_prob = float(probabilities[frame_idx, string_idx, silence_class])
                if prob - silence_prob <= penalty:
                    decoded[frame_idx, string_idx] = silence_class

    return decoded


def decode_variant(
    probabilities: np.ndarray,
    profile: tools.GuitarProfile,
    variant: DecodeVariant,
) -> np.ndarray:
    probs = np.asarray(probabilities)
    silence_class = probs.shape[-1] - 1
    if variant.name == "baseline_top1":
        return decode_baseline_top1(probs)

    decoded_by_frame = np.zeros((probs.shape[0], probs.shape[1]), dtype=np.int64)
    for string_idx in range(probs.shape[1]):
        decoded_by_frame[:, string_idx] = viterbi_decode_string(
            probs[:, string_idx, :],
            top_k=variant.top_k,
            transition_penalty=variant.transition_penalty,
            silence_class=silence_class,
        )

    decoded_by_frame = suppress_duplicate_pitches(
        decoded_by_frame,
        probs,
        profile,
        penalty=variant.duplicate_pitch_penalty,
    )
    decoded_by_frame[decoded_by_frame == silence_class] = -1
    return decoded_by_frame.T


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_model_for_checkpoint(config_path: Path, checkpoint_path: Path, device_override: str) -> tuple[Any, Any, Any, Any, Any]:
    cfg = load_config(config_path)
    cfg.runtime.device = device_override
    device = resolve_device(cfg.runtime)
    data_proc = build_feature_extractor(cfg)
    profile = build_profile()
    estimator, _ = build_estimators(profile)
    model = build_model(cfg, data_proc, profile, device)

    payload = safe_torch_load(checkpoint_path, device)
    if not isinstance(payload, dict) or payload.get("checkpoint_type") != "tabcnn_synthtab_training_state":
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if hasattr(model, "change_device"):
        model.change_device(device)
    if "model_iter" in payload:
        model.iter = int(payload["model_iter"])
    model.eval()
    return cfg, data_proc, profile, estimator, model


def evaluate_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    selected_tracks: list[TrackCandidate],
    device_override: str,
) -> dict[str, Any]:
    cfg, data_proc, profile, estimator, model = load_model_for_checkpoint(config_path, checkpoint_path, device_override)
    dataset = create_synthtab_dataset(cfg, "val", data_proc, profile)
    # Full-track diagnostics avoid random 500-frame validation windows and make
    # checkpoint-to-checkpoint comparisons deterministic for the selected tracks.
    dataset.seq_length = None
    dataset.num_frames = None
    available = set(dataset.tracks)
    missing = [track.track_id for track in selected_tracks if track.track_id not in available]
    if missing:
        raise ValueError(f"{len(missing)} selected tracks are not present in the active validation set. First missing: {missing[0]}")

    aggregate_counts = Counts()
    track_reports: list[TrackReport] = []
    with torch.no_grad():
        for index, candidate in enumerate(selected_tracks, start=1):
            print(f"[{checkpoint_path.name}] {index}/{len(selected_tracks)} {candidate.bucket}: {candidate.track_id}", flush=True)
            track_data = dataset.get_track_data(candidate.track_id)
            predictions = run_offline(track_data, model, estimator)
            counts = compare_tablature(track_data[tools.KEY_TABLATURE], predictions[tools.KEY_TABLATURE], profile)
            aggregate_counts.add(counts)
            metrics = summarize_counts(counts)
            track_reports.append(
                TrackReport(
                    track_id=candidate.track_id,
                    bucket=candidate.bucket,
                    source_multi_pitch_f1=candidate.multi_pitch_f1,
                    source_tablature_f1=candidate.tablature_f1,
                    source_gap=candidate.gap,
                    source_ref_silence_ratio=candidate.ref_silence_ratio,
                    counts=counts,
                    metrics=metrics,
                )
            )

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "model_iter": int(getattr(model, "iter", -1)),
        "tracks": [track_report_to_dict(report) for report in track_reports],
        "aggregate": summarize_counts(aggregate_counts),
    }


def evaluate_topk_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    selected_tracks: list[TrackCandidate],
    device_override: str,
) -> dict[str, Any]:
    cfg, data_proc, profile, _estimator, model = load_model_for_checkpoint(config_path, checkpoint_path, device_override)
    dataset = create_synthtab_dataset(cfg, "val", data_proc, profile)
    dataset.seq_length = None
    dataset.num_frames = None
    available = set(dataset.tracks)
    missing = [track.track_id for track in selected_tracks if track.track_id not in available]
    if missing:
        raise ValueError(f"{len(missing)} selected tracks are not present in the active validation set. First missing: {missing[0]}")

    aggregate_counts = TopKCounts()
    track_reports: list[TopKTrackReport] = []
    with torch.no_grad():
        for index, candidate in enumerate(selected_tracks, start=1):
            print(f"[topk:{checkpoint_path.name}] {index}/{len(selected_tracks)} {candidate.bucket}: {candidate.track_id}", flush=True)
            track_data = dataset.get_track_data(candidate.track_id)
            predicted, probabilities = run_raw_tablature(track_data, model)
            counts = analyze_topk(track_data[tools.KEY_TABLATURE], predicted, probabilities, profile)
            aggregate_counts.add(counts)
            metrics = summarize_topk_counts(counts)
            track_reports.append(
                TopKTrackReport(
                    track_id=candidate.track_id,
                    bucket=candidate.bucket,
                    source_multi_pitch_f1=candidate.multi_pitch_f1,
                    source_tablature_f1=candidate.tablature_f1,
                    source_gap=candidate.gap,
                    source_ref_silence_ratio=candidate.ref_silence_ratio,
                    counts=counts,
                    metrics=metrics,
                )
            )

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "model_iter": int(getattr(model, "iter", -1)),
        "num_groups": int(model.dense[-1].num_groups),
        "num_classes": int(model.dense[-1].num_classes),
        "silence_class": int(model.dense[-1].num_classes - 1),
        "tracks": [topk_track_report_to_dict(report) for report in track_reports],
        "aggregate": summarize_topk_counts(aggregate_counts),
    }


def evaluate_decode_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    selected_tracks: list[TrackCandidate],
    device_override: str,
) -> dict[str, Any]:
    cfg, data_proc, profile, _estimator, model = load_model_for_checkpoint(config_path, checkpoint_path, device_override)
    dataset = create_synthtab_dataset(cfg, "val", data_proc, profile)
    dataset.seq_length = None
    dataset.num_frames = None
    available = set(dataset.tracks)
    missing = [track.track_id for track in selected_tracks if track.track_id not in available]
    if missing:
        raise ValueError(f"{len(missing)} selected tracks are not present in the active validation set. First missing: {missing[0]}")

    aggregate_counts = {variant.name: Counts() for variant in DECODE_VARIANTS}
    per_track_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for index, candidate in enumerate(selected_tracks, start=1):
            print(f"[decode:{checkpoint_path.name}] {index}/{len(selected_tracks)} {candidate.bucket}: {candidate.track_id}", flush=True)
            track_data = dataset.get_track_data(candidate.track_id)
            _predicted, probabilities = run_raw_tablature(track_data, model)
            reference = track_data[tools.KEY_TABLATURE]

            for variant in DECODE_VARIANTS:
                decoded = decode_variant(probabilities, profile, variant)
                counts = compare_tablature(reference, decoded, profile)
                aggregate_counts[variant.name].add(counts)
                per_track_rows.append(
                    {
                        "track_id": candidate.track_id,
                        "bucket": candidate.bucket,
                        "variant": variant.name,
                        "source_multi_pitch_f1": candidate.multi_pitch_f1,
                        "source_tablature_f1": candidate.tablature_f1,
                        "source_gap": candidate.gap,
                        "source_ref_silence_ratio": candidate.ref_silence_ratio,
                        **summarize_counts(counts),
                    }
                )

    variant_reports = []
    baseline = summarize_counts(aggregate_counts["baseline_top1"])
    baseline_extra = baseline["extra_share_of_pred"]
    for variant in DECODE_VARIANTS:
        metrics = summarize_counts(aggregate_counts[variant.name])
        exact_delta = float(metrics["exact_f1"] - baseline["exact_f1"])
        pitch_delta = float(metrics["pitch_f1"] - baseline["pitch_f1"])
        extra_delta = float(metrics["extra_share_of_pred"] - baseline_extra)
        extra_relative_delta = float(extra_delta / baseline_extra) if baseline_extra else 0.0
        passes = (
            variant.name != "baseline_top1"
            and exact_delta >= 0.01
            and pitch_delta >= -0.01
            and extra_relative_delta <= 0.05
        )
        variant_reports.append(
            {
                "variant": variant.name,
                "top_k": variant.top_k,
                "transition_penalty": variant.transition_penalty,
                "duplicate_pitch_penalty": variant.duplicate_pitch_penalty,
                **metrics,
                "exact_f1_delta": exact_delta,
                "pitch_f1_delta": pitch_delta,
                "extra_share_delta": extra_delta,
                "extra_share_relative_delta": extra_relative_delta,
                "passes_acceptance": passes,
            }
        )

    candidates = [variant for variant in variant_reports if variant["passes_acceptance"]]
    best = max(candidates, key=lambda item: item["exact_f1_delta"], default=None)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "model_iter": int(getattr(model, "iter", -1)),
        "variants": variant_reports,
        "tracks": per_track_rows,
        "best_variant": best["variant"] if best else None,
        "accepted": best is not None,
    }


def track_report_to_dict(report: TrackReport) -> dict[str, Any]:
    return {
        "track_id": report.track_id,
        "bucket": report.bucket,
        "source_multi_pitch_f1": report.source_multi_pitch_f1,
        "source_tablature_f1": report.source_tablature_f1,
        "source_gap": report.source_gap,
        "source_ref_silence_ratio": report.source_ref_silence_ratio,
        **report.metrics,
    }


def topk_track_report_to_dict(report: TopKTrackReport) -> dict[str, Any]:
    return {
        "track_id": report.track_id,
        "bucket": report.bucket,
        "source_multi_pitch_f1": report.source_multi_pitch_f1,
        "source_tablature_f1": report.source_tablature_f1,
        "source_gap": report.source_gap,
        "source_ref_silence_ratio": report.source_ref_silence_ratio,
        **report.metrics,
    }


def write_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint_report in reports:
        checkpoint_name = checkpoint_report["checkpoint_name"]
        model_iter = checkpoint_report["model_iter"]
        for track in checkpoint_report["tracks"]:
            rows.append({"checkpoint": checkpoint_name, "model_iter": model_iter, **track})
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, report: dict[str, Any], suffix: str = "") -> None:
    suffix_part = f".{suffix}" if suffix else ""
    write_json(output_dir / f"aggregate{suffix_part}.json", report)
    write_csv(output_dir / f"per_track{suffix_part}.csv", report["checkpoints"])
    write_json(output_dir / f"per_track{suffix_part}.json", report["checkpoints"])
    (output_dir / f"summary{suffix_part}.md").write_text(render_summary(report), encoding="utf-8")


def write_topk_outputs(output_dir: Path, report: dict[str, Any], suffix: str = "") -> None:
    suffix_part = f".{suffix}" if suffix else ""
    write_json(output_dir / f"aggregate{suffix_part}.json", report)
    write_csv(output_dir / f"per_track{suffix_part}.csv", report["checkpoints"])
    write_json(output_dir / f"per_track{suffix_part}.json", report["checkpoints"])
    (output_dir / f"summary{suffix_part}.md").write_text(render_topk_summary(report), encoding="utf-8")


def write_decode_outputs(output_dir: Path, report: dict[str, Any], suffix: str = "") -> None:
    suffix_part = f".{suffix}" if suffix else ""
    write_json(output_dir / f"aggregate{suffix_part}.json", report)
    write_decode_csv(output_dir / f"per_track{suffix_part}.csv", report["checkpoints"])
    write_json(output_dir / f"per_track{suffix_part}.json", report["checkpoints"])
    (output_dir / f"summary{suffix_part}.md").write_text(render_decode_summary(report), encoding="utf-8")


def write_decode_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint_report in reports:
        checkpoint_name = checkpoint_report["checkpoint_name"]
        model_iter = checkpoint_report["model_iter"]
        for track in checkpoint_report["tracks"]:
            rows.append({"checkpoint": checkpoint_name, "model_iter": model_iter, **track})
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_summary(report: dict[str, Any]) -> str:
    lines = [
        "# String/Fret Frame Diagnostic",
        "",
        f"- Config: `{report['config']}`",
        f"- Track source experiment: `{report['track_source_experiment']}`",
        f"- Selected tracks: `{len(report['selected_tracks'])}`",
        "",
        "| Checkpoint | Exact F1 | Pitch F1 | Oracle Tab F1 | Canonical F1 | Wrong-position/ref | Missed/ref | Extra/pred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint_report in report["checkpoints"]:
        agg = checkpoint_report["aggregate"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{checkpoint_report['checkpoint_name']}`",
                    pct(agg["exact_f1"]),
                    pct(agg["pitch_f1"]),
                    pct(agg["oracle_tab_f1"]),
                    pct(agg["canonical_f1"]),
                    pct(agg["wrong_position_share_of_ref"]),
                    pct(agg["missed_share_of_ref"]),
                    pct(agg["extra_share_of_pred"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `Exact F1` measures exact string/fret agreement.",
            "- `Pitch F1` ignores string/fret placement and scores only the pitch multiset.",
            "- `Oracle Tab F1` is the upper bound if every pitch-correct but position-wrong note were moved to the reference position.",
            "- `Canonical F1` applies a deterministic pitch-to-position rule without logits; if it is worse than exact F1, hard post-processing is not useful.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_decode_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Top-K Constrained Decoding Experiment",
        "",
        f"- Config: `{report['config']}`",
        f"- Track source experiment: `{report['track_source_experiment']}`",
        f"- Selected tracks: `{len(report['selected_tracks'])}`",
        "",
        "| Variant | Exact F1 | Pitch F1 | Wrong-position/ref | Missed/ref | Extra/pred | Exact delta | Pitch delta | Extra rel delta | Passes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for checkpoint_report in report["checkpoints"]:
        for variant in checkpoint_report["variants"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{variant['variant']}`",
                        pct(variant["exact_f1"]),
                        pct(variant["pitch_f1"]),
                        pct(variant["wrong_position_share_of_ref"]),
                        pct(variant["missed_share_of_ref"]),
                        pct(variant["extra_share_of_pred"]),
                        pct(variant["exact_f1_delta"]),
                        pct(variant["pitch_f1_delta"]),
                        pct(variant["extra_share_relative_delta"]),
                        str(variant["passes_acceptance"]).lower(),
                    ]
                )
                + " |"
            )
        lines.append("")
        if checkpoint_report["best_variant"]:
            lines.append(f"Accepted best variant: `{checkpoint_report['best_variant']}`.")
        else:
            lines.append("No variant passed acceptance criteria.")

    lines.extend(
        [
            "",
            "## Acceptance Criteria",
            "",
            "- Exact string/fret F1 must improve by at least `+1 pp` over baseline.",
            "- Pitch F1 must not drop by more than `1 pp`.",
            "- Extra/pred must not increase by more than `5%` relative.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_topk_summary(report: dict[str, Any]) -> str:
    class_note = "unknown"
    if report["checkpoints"]:
        first_checkpoint = report["checkpoints"][0]
        class_note = (
            f"{first_checkpoint.get('num_groups')} string groups x "
            f"{first_checkpoint.get('num_classes')} classes "
            f"({first_checkpoint.get('silence_class')} is silence)"
        )

    lines = [
        "# Logits-Aware Top-K String/Fret Diagnostic",
        "",
        f"- Config: `{report['config']}`",
        f"- Track source experiment: `{report['track_source_experiment']}`",
        f"- Selected tracks: `{len(report['selected_tracks'])}`",
        f"- Output layout: `{class_note}`",
        "",
        "| Checkpoint | Correct top-1 | Correct top-3 | Correct top-5 | Pitch-compatible top-1 | Pitch-compatible top-3 | Pitch-compatible top-5 | PCW correct top-5 | PCW pitch-compatible top-5 | Avg PCW top1-correct prob | Extra weak <=10% | Avg extra margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint_report in report["checkpoints"]:
        agg = checkpoint_report["aggregate"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{checkpoint_report['checkpoint_name']}`",
                    pct(agg["correct_top1_rate"]),
                    pct(agg["correct_top3_rate"]),
                    pct(agg["correct_top5_rate"]),
                    pct(agg["pitch_compatible_top1_rate"]),
                    pct(agg["pitch_compatible_top3_rate"]),
                    pct(agg["pitch_compatible_top5_rate"]),
                    pct(agg["pcw_correct_top5_rate"]),
                    pct(agg["pcw_pitch_compatible_top5_rate"]),
                    pct(agg["avg_pcw_top1_minus_correct_prob"]),
                    pct(agg["weak_extra_margin_010_rate"]),
                    pct(agg["avg_extra_note_margin"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `Correct top-k` ranks the exact reference string/fret inside that string's softmax classes.",
            "- `Pitch-compatible top-k` ranks the best valid fretboard position for the same MIDI pitch among all non-silence string/fret classes.",
            "- `PCW` means pitch-correct but string/fret-wrong notes from top-1 predictions.",
            "- `Avg PCW top1-correct prob` is the probability gap between the chosen top-1 class and the reference string/fret class.",
            "- `Extra weak <=10%` estimates how often extra notes have weak note-vs-silence confidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--track-source-experiment", type=Path, required=True)
    parser.add_argument("--max-tracks", type=int, default=20)
    parser.add_argument("--max-ref-silence", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="List selected tracks without loading model/checkpoints.")
    parser.add_argument("--topk", action="store_true", help="Run logits-aware top-k string/fret diagnostics.")
    parser.add_argument("--decode-experiment", action="store_true", help="Run top-k constrained decoding experiment.")
    args = parser.parse_args()
    if args.topk and args.decode_experiment:
        raise SystemExit("--topk and --decode-experiment are mutually exclusive.")

    root = Path.cwd()
    config_path = resolve_path(args.config, root)
    source_experiment = resolve_path(args.track_source_experiment, root)
    if args.decode_experiment:
        default_output_dir = Path("generated/diagnostics/string_fret_constrained_lespaul_20tracks")
    elif args.topk:
        default_output_dir = Path("generated/diagnostics/string_fret_topk_lespaul_20tracks")
    else:
        default_output_dir = Path("generated/diagnostics/string_fret_lespaul_20tracks")
    output_dir = resolve_path(args.output_dir or default_output_dir, root)
    selected_tracks = load_track_candidates(source_experiment, args.max_tracks, args.max_ref_silence)

    if args.dry_run:
        for track in selected_tracks:
            print(
                f"{track.bucket}\t{pct(track.multi_pitch_f1)}\t{pct(track.tablature_f1)}\t"
                f"{pct(track.gap)}\tref_silence={pct(track.ref_silence_ratio)}\t{track.track_id}"
            )
        return

    if not args.checkpoint:
        raise SystemExit("At least one --checkpoint is required unless --dry-run is used.")

    checkpoints = [resolve_path(checkpoint, root) for checkpoint in args.checkpoint]
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    if args.topk:
        checkpoint_reports = []
        report = {
            "mode": "topk",
            "config": str(config_path),
            "track_source_experiment": str(source_experiment),
            "selected_tracks": [track.__dict__ for track in selected_tracks],
            "checkpoints": checkpoint_reports,
        }

        for checkpoint in checkpoints:
            checkpoint_reports.append(evaluate_topk_checkpoint(config_path, checkpoint, selected_tracks, args.device))
            write_topk_outputs(output_dir, report, suffix="partial")

        write_topk_outputs(output_dir, report)
        print(f"Wrote diagnostics to {output_dir}")
        return

    if args.decode_experiment:
        checkpoint_reports = []
        report = {
            "mode": "decode",
            "config": str(config_path),
            "track_source_experiment": str(source_experiment),
            "selected_tracks": [track.__dict__ for track in selected_tracks],
            "checkpoints": checkpoint_reports,
        }

        for checkpoint in checkpoints:
            checkpoint_reports.append(evaluate_decode_checkpoint(config_path, checkpoint, selected_tracks, args.device))
            write_decode_outputs(output_dir, report, suffix="partial")

        write_decode_outputs(output_dir, report)
        print(f"Wrote diagnostics to {output_dir}")
        return

    checkpoint_reports = []
    report = {
        "mode": "frame",
        "config": str(config_path),
        "track_source_experiment": str(source_experiment),
        "selected_tracks": [track.__dict__ for track in selected_tracks],
        "checkpoints": checkpoint_reports,
    }

    for checkpoint in checkpoints:
        checkpoint_reports.append(evaluate_checkpoint(config_path, checkpoint, selected_tracks, args.device))
        write_outputs(output_dir, report, suffix="partial")

    write_outputs(output_dir, report)
    print(f"Wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()

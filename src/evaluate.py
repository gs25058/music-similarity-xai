from pathlib import Path
import json

import numpy as np
from scipy.stats import rankdata

from .compare import analyze_song, compare_songs
from .data import save_json
from .features import FACTORS
from .judge import judge_pair, swap_comparison
from .train import audit_manifest


def binary_metrics(targets, scores, threshold=.8):
    y, s = np.asarray(targets), np.asarray(scores)
    if not len(y):
        return dict(status="unavailable", n=0)
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    auc = float(((rankdata(s)[y == 1].sum()-positives*(positives+1)/2)/(positives*negatives))) if positives and negatives else None
    predicted = s >= threshold
    return dict(status="evaluated", n=len(y), positives=positives, negatives=negatives, roc_auc=auc,
                accuracy=float(np.mean(predicted == y)), threshold=threshold,
                false_positive_rate=float(np.mean(predicted[y == 0])) if negatives else None,
                positive_mean_cosine=float(s[y == 1].mean()) if positives else None,
                negative_mean_cosine=float(s[y == 0].mean()) if negatives else None)


def interval_iou(a, b):
    intersection = max(0, min(a["end"], b["end"])-max(a["start"], b["start"]))
    union = max(a["end"], b["end"])-min(a["start"], b["start"])
    return intersection/union if union else 0.


def evaluate(manifest_path, backend, out, split="test", threshold=.8, max_pairs=0,
             judge="rule", qwen_python=None, device="cpu", judge_cases=2):
    manifest = json.loads(Path(manifest_path).read_text())
    audit_manifest(manifest)
    pairs = [p for p in manifest["pairs"] if p["split"] == split]
    if max_pairs:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError(f"No pairs in {split}")
    cache, measurements = {}, {f: ([], []) for f in backend.supported}
    hard = []

    def analysis(path):
        if path not in cache:
            cache[path] = analyze_song(path, backend)
        return cache[path]

    for p in pairs:
        comparison = compare_songs(analysis(p["a"]["audio"]), analysis(p["b"]["audio"]), backend, threshold=threshold)
        for factor in backend.supported:
            # MERIT melody is an explicitly approximate comparison to interval labels.
            target_key = "pitch_intervals" if factor == "melody" else factor
            target, score = p["targets"].get(target_key), comparison["matrix_maxima"].get(factor)
            if target is not None and score is not None:
                measurements[factor][0].append(target)
                measurements[factor][1].append(score)
        if p["kind"] == "hard_negative":
            pitch_key = "melody" if backend.name == "merit" else "pitch_intervals"
            score = comparison["matrix_maxima"].get(pitch_key)
            hard.append(dict(pitch_cosine=score, pitch_false_positive=score >= threshold if score is not None else None,
                             overall_rule_judgment=judge_pair(comparison)["selected"]["judgment"]))
    insertions, judgments = [], []
    for case in [p for p in manifest.get("insertion_cases", []) if p["split"] == split]:
        comparison = compare_songs(analysis(case["a"]), analysis(case["b"]), backend, threshold=threshold)
        candidates = comparison["evidence"]
        matches = [(min(interval_iou(e["a"], case["truth"]["a"]), interval_iou(e["b"], case["truth"]["b"])), e) for e in candidates]
        best_iou, best = max(matches, key=lambda pair: pair[0], default=(0., None))
        top = candidates[0] if candidates else None
        insertions.append(dict(truth=case["truth"], best_bilateral_min_iou=best_iou,
                               detected_both_iou_05=best_iou >= .5,
                               top1_bilateral_min_iou=min(interval_iou(top["a"], case["truth"]["a"]), interval_iou(top["b"], case["truth"]["b"])) if top else 0.,
                               best_evidence=best,
                               boundary_absolute_error={side: {key: abs(best[side][key]-case["truth"][side][key]) for key in ("start", "end")}
                                                        for side in ("a", "b")} if best else None))
        if len(judgments) < judge_cases:
            original = judge_pair(comparison, judge, qwen_python, device)
            swapped = judge_pair(swap_comparison(comparison), judge, qwen_python, device)
            success = all(v["execution"]["status"] in ("executed", "executed_validated") for v in (original, swapped))
            judgments.append(dict(original=original, swapped=swapped, both_executed=success,
                                  id_validation_passed=success,
                                  swap_judgment_consistent=(original["selected"]["judgment"] == swapped["selected"]["judgment"]) if success else None,
                                  swap_evidence_ids_consistent=(set(original["selected"]["evidence_ids"]) == set(swapped["selected"]["evidence_ids"])) if success else None))
    successful = [c for c in judgments if c["both_executed"]]
    responses = [c[side] for c in judgments for side in ("original", "swapped")]
    first_pass = [v["execution"].get("metadata", {}).get("first_pass_valid", False) for v in responses]
    valid_hard_negatives = [v["pitch_false_positive"] for v in hard if v["pitch_false_positive"] is not None]
    result = dict(backend=backend.metadata(), split=split, pair_count=len(pairs),
                  factor_metrics={factor: binary_metrics(*values, threshold) for factor, values in measurements.items()},
                  merit_comparison=dict(strict_common_factor="rhythm", approximate_common_factor="melody vs pitch_intervals (not identical)",
                                        excluded=["timbre vs accompaniment_bass", "harmony", "phrase_structure"]),
                  insertion_localization=insertions, hard_negatives=hard,
                  hard_negative_pitch_fpr=float(np.mean(valid_hard_negatives)) if valid_hard_negatives else None,
                  hard_negative_overall_high_similarity_rate=float(np.mean([v["overall_rule_judgment"] == "high_similarity" for v in hard])) if hard else None,
                  judge_backend=judge, judge_checks=judgments,
                  judge_metrics=dict(requested_responses=len(responses),
                      known_id_and_schema_pass_rate=float(np.mean([v["execution"]["status"] in ("executed", "executed_validated") for v in responses])) if responses else None,
                      first_pass_schema_pass_rate=float(np.mean(first_pass)) if responses and judge == "qwen" else None,
                      swap_judgment_consistency=float(np.mean([c["swap_judgment_consistent"] for c in successful])) if successful else None,
                      swap_evidence_id_consistency=float(np.mean([c["swap_evidence_ids_consistent"] for c in successful])) if successful else None,
                      swap_valid_and_consistent_rate=float(np.mean([c["both_executed"] and c["swap_judgment_consistent"] for c in judgments])) if judgments else None,
                      note="Fallback rule outputs are excluded from Qwen consistency; validation is schema/ID integrity, not proof of explanation faithfulness."),
                  note="Threshold supplied, not tuned on test. Small synthetic test is not real-world performance.")
    save_json(out, result)
    return result

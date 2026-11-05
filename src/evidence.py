"""Ordered, source-aware evidence. Scores and analyzer reliability stay separate."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

FACTOR_NAMES = {"pitch_intervals": "음정 진행", "rhythm": "리듬", "harmony": "화성",
                "accompaniment_bass": "반주·베이스", "melody": "선율", "timbre": "음색",
                "phrase_structure": "프레이즈 구조"}
SCORE_DEFINITION = {"name": "cosine", "range": [-1, 1], "direction": "larger_is_more_similar",
                    "meaning": "mean cosine of corresponding window embeddings; not a probability"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def temporal_observations(times, pitch=None, bass=None, onset=None, beat=None, chroma=None,
                          chord_root=None, chord_quality=None, source="prediction", max_bins=12,
                          melody_onset=None,bass_onset=None):
    """Contiguous bins (not unordered histograms), always in original song time.

    Pitch inputs are MIDI with NaN for unvoiced; onset/beat are probabilities
    or normalized proxy strengths, never silently labeled detected ground truth.
    """
    times = np.asarray(times)
    if not len(times):return dict(status="unavailable",source=source,sequence=[],onsets=dict(status="unavailable"),reason="no_valid_frames")
    groups = np.array_split(np.arange(len(times)), min(max_bins, len(times)))
    sequence = []
    for indices in groups:
        if not len(indices):
            continue
        item = dict(time=round(float(times[indices[0]]), 3))
        for key, values in (("pitch_midi", pitch), ("bass_midi", bass)):
            if values is not None:
                segment = np.asarray(values, dtype=float)[indices]
                voiced = segment[np.isfinite(segment)]
                item[key] = round(float(np.median(voiced)), 2) if len(voiced) else None
        for key, values in (("onset_strength", onset), ("beat_strength", beat),
                            ("melody_onset_strength",melody_onset),("bass_onset_strength",bass_onset)):
            if values is not None:
                item[key] = round(float(np.max(np.asarray(values)[indices])), 3)
        if chroma is not None:
            item["chroma"] = np.asarray(chroma)[indices].mean(0).round(3).tolist()
        for key, values in (("chord_root", chord_root), ("chord_quality", chord_quality)):
            if values is not None:
                v, counts = np.unique(np.asarray(values)[indices], return_counts=True)
                item[key] = int(v[np.argmax(counts)])
        sequence.append(item)
    for key in ("pitch_midi", "bass_midi"):
        previous = None
        previous_time=None
        for item in sequence:
            value = item.get(key)
            if value is not None and previous is not None:
                item[key.replace("_midi", "_interval")] = round(value - previous, 2)
                item[key.replace("_midi","_interval_reference")]=dict(time=previous_time,midi=previous,relation="previous_original_bin")
            previous = value
            previous_time=item["time"]
    event_indices = []
    if onset is not None and len(times):
        values = np.asarray(onset)
        event_indices = [i for i in range(1, len(values) - 1)
                         if values[i] >= .5 and values[i] > values[i-1] and values[i] >= values[i+1]]
    event_times = times[event_indices]
    events = dict(status="available" if onset is not None else "unavailable",
                  times_seconds=np.round(event_times, 3).tolist(), source=source,
                  threshold=.5, threshold_status="uncalibrated")
    if len(event_times) > 1:
        durations = np.diff(event_times)
        events["relative_inter_onset_durations"] = np.round(durations / max(float(np.median(durations)), 1e-6), 3).tolist()
    # Only an explicit beat prediction/measurement permits beat-normalized IOIs.
    beat_peaks = []
    if beat is not None:
        v = np.asarray(beat)
        beat_peaks = [i for i in range(1, len(v)-1) if v[i] >= .5 and v[i] > v[i-1] and v[i] >= v[i+1]]
    if len(beat_peaks) >= 2:
        beat_times = times[beat_peaks]
        period = float(np.median(np.diff(beat_times)))
        events["beat_normalized"] = dict(status="predicted_not_verified", period_seconds=round(period, 3),
            positions=np.round((event_times-beat_times[0])/period, 3).tolist())
    else:
        events["beat_normalized"] = dict(status="unavailable", reason="insufficient_beat_peaks")
    result=dict(status="available", source=source, representation="contiguous_time_bins_v1",
                resolution="bin summaries; not note transcription", sequence=sequence, onsets=events,
                availability={k:"available" if v is not None else "unavailable" for k,v in
                    dict(pitch=pitch,bass=bass,onset=onset,beat=beat,chroma=chroma,chord_root=chord_root,chord_quality=chord_quality,
                         melody_onset=melody_onset,bass_onset=bass_onset).items()})
    for name,values in (("melody_onsets",melody_onset),("bass_onsets",bass_onset)):
        result[name]=temporal_observations(times,onset=values,beat=beat,source=source,max_bins=max_bins)["onsets"] if values is not None else dict(status="unavailable")
    return result


def _iou(a, b):
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return max(0., min(a["end"], b["end"]) - max(a["start"], b["start"])) / max(union, 1e-9)


def representative_evidence(items, limit=3, overlap=.6):
    """NMS requires overlap on BOTH songs; preserve IDs and suppressed aliases."""
    selected, aliases = [], {}
    for item in sorted(items, key=lambda e: -e["combined_cosine"]):
        duplicate = next((e for e in selected if min(_iou(e["a"], item["a"]), _iou(e["b"], item["b"])) >= overlap), None)
        if duplicate:
            aliases.setdefault(duplicate["evidence_id"], []).append(item["evidence_id"])
        elif len(selected) < limit:
            selected.append(item)
    return selected, aliases


def _ordered_info(e, side, songs):
    result = []
    for n, info in enumerate(e.get("music_information", {}).get(side, [])):
        window = None
        pairs = e.get("window_pairs", [])
        if songs and n < len(pairs):
            index = pairs[n][side]
            window = next((w for w in songs[side]["windows"] if w["index"] == index), None)
        temporal = info.get("temporal", dict(status="unavailable", reason="historical_report_has_no_ordered_frames"))
        record = dict(window_index=window["index"] if window else None,
                      interval={k: window[k] for k in ("start", "end")} if window else copy.deepcopy(e[side]),
                      provenance={k: copy.deepcopy(info[k]) for k in ("status", "source", "pitch_source", "harmony_source",
                          "accompaniment_source", "training_accompaniment_sources", "missing") if k in info},
                      temporal=copy.deepcopy(temporal))
        if temporal["status"] == "unavailable":
            # Preserve each historical window summary, never manufacture order
            # within an old median/class histogram.
            from .judge import music_information_summary
            record["window_summary"] = music_information_summary(info)
        result.append(record)
    return sorted(result, key=lambda x: x["interval"]["start"])


def sequence_comparison(ordered):
    """Compare corresponding bins per paired window, explicitly approximate."""
    scores = {f: [] for f in ("pitch_intervals", "rhythm", "harmony", "accompaniment_bass")}
    for wa, wb in zip(ordered["a"], ordered["b"]):
        a, b = [w["temporal"].get("sequence", []) for w in (wa, wb)]
        if min(len(a),len(b)) < 3:
            continue
        n = min(len(a),len(b))
        a = [a[int(i)] for i in np.linspace(0,len(a)-1,n)]
        b = [b[int(i)] for i in np.linspace(0,len(b)-1,n)]
        for factor, key in (("pitch_intervals","pitch_interval"),("accompaniment_bass","bass_interval")):
            valid = [(x.get(key),y.get(key)) for x,y in zip(a,b) if x.get(key) is not None and y.get(key) is not None]
            if len(valid) >= 2:
                # A bounded derived agreement, not a probability or note match.
                scores[factor].append(float(np.mean([max(0.,1-abs(x-y)/12) for x,y in valid])))
        for factor, key in (("rhythm","onset_strength"),("harmony","chroma")):
            values = [(x.get(key),y.get(key)) for x,y in zip(a,b) if key in x and key in y]
            if len(values)>=3:
                x,y = (np.asarray(v,dtype=float).reshape(-1) for v in zip(*values))
                denom=float(np.linalg.norm(x)*np.linalg.norm(y))
                if denom>1e-8:
                    scores[factor].append(float(x@y/denom))
    return {factor: dict(status="available" if values else "unavailable",
        derived_agreement=round(float(np.mean(values)),4) if values else None, n_windows=len(values),
        method="paired_window_normalized_time_bins; interval_absolute_difference/12 or cosine",
        calibration_status="uncalibrated", interpretation_threshold=.8,
        scope="approximate predicted/proxy bin patterns; not verified note alignment") for factor,values in scores.items()}


def compact_evidence(comparison, backend=None, songs=None, calibration=None, limit=3, time_bins=6):
    backend = copy.deepcopy(backend or comparison.get("backend_identity", {}))
    selected, aliases = representative_evidence(comparison["evidence"], limit)
    items = []
    for e in selected:
        ordered = {side: _ordered_info(e, side, songs) for side in ("a", "b")}
        for side in ("a", "b"):
            for record in ordered[side]:
                seq=record["temporal"].get("sequence",[])
                # A retained interval refers to its original neighboring bin,
                # which may be omitted by subsampling. Keep that neighbor's
                # value/time so it cannot be mistaken for the retained gap.
                for i,item in enumerate(seq):
                    item["original_bin_index"]=i
                    if i:
                        for key in ("pitch","bass"):
                            previous=seq[i-1].get(key+"_midi");value=item.get(key+"_midi")
                            if previous is not None and value is not None:
                                item[key+"_interval"]=round(value-previous,2)
                                item[key+"_interval_reference"]=dict(time=seq[i-1]["time"],midi=previous,relation="previous_original_bin")
                if len(seq)>time_bins:
                    record["temporal"]["original_bins"]=len(seq)
                    record["temporal"]["sampling"]="ordered_evenly_spaced_bins_including_endpoints"
                    record["temporal"]["sequence"]=[seq[int(i)] for i in np.linspace(0,len(seq)-1,time_bins)]
        raw_sequence = e.get("alignment", [])
        alignment = dict(status="available" if raw_sequence else "unavailable",
                         method="monotone_window_pairs; no note-level alignment", windows=copy.deepcopy(raw_sequence),
                         duration_seconds={side: round(e[side]["end"]-e[side]["start"], 3) for side in ("a", "b")},
                         matched_windows=len(e.get("window_pairs", [])),
                         note_alignment=dict(status="unavailable"))
        factors = {}
        for name, value in e["factors"].items():
            available = value.get("cosine") is not None and value.get("status") != "unavailable"
            factor_cal = dict(status="uncalibrated", method=None, n=0)
            if calibration and name in calibration.get("factors", {}):
                identity = calibration.get("backend_sha256")
                if identity != digest(backend):
                    raise ValueError("Calibration backend/checkpoint identity mismatch")
                factor_cal = copy.deepcopy(calibration["factors"][name])
                if calibration.get("checkpoint_sha256") and backend.get("checkpoint"):
                    if file_digest(backend["checkpoint"])!=calibration["checkpoint_sha256"]:raise ValueError("Calibration checkpoint content mismatch")
                if factor_cal.get("score_unit")!="aligned_window_pair_cosine" or len(e.get("window_pairs",[]))!=1:
                    factor_cal=dict(status="uncalibrated",n=0,method=None,
                        reason="validation_score_unit_does_not_match_candidate",available_validation_fit=factor_cal)
            source_statuses = sorted({v["provenance"].get("status", "unknown") for side in ("a", "b") for v in ordered[side]})
            observed = [a["factors"].get(name) for a in raw_sequence if a.get("factors", {}).get(name) is not None]
            threshold = factor_cal.get("threshold", .8)
            factors[name] = dict(status="available" if available else "unavailable", name_ko=FACTOR_NAMES[name],
                raw_cosine=float(value["cosine"]) if available else None, score_definition_ref="score_definition",
                head=dict(name=name, backend=backend.get("backend", "unknown"), checkpoint=backend.get("checkpoint"),
                          checkpoint_sha256=backend.get("checkpoint_sha256"),
                          backbone_revision=backend.get("training", {}).get("model_versions", {}).get("mert95", {}).get("revision")),
                provenance=dict(kinds=source_statuses, status="missing" if not available else "prediction" if "model_prediction_not_verified" in source_statuses else "derived_measurement"),
                analyzer_reliability=dict(status="not_validated", n=0, note="Similarity magnitude is not analyzer reliability"),
                calibration=factor_cal,
                interpretation_threshold=threshold, threshold_status="validation_fitted" if factor_cal.get("status", "").startswith("fitted") else "uncalibrated",
                persistence=dict(status="available" if observed else "unavailable", n_windows=len(observed),
                    above_threshold=sum(v >= threshold for v in observed),
                    mismatches=[i for i, v in enumerate(observed) if v < threshold]))
        items.append(dict(evidence_id=e["evidence_id"], aliases=aliases.get(e["evidence_id"], []),
                          a=copy.deepcopy(e["a"]), b=copy.deepcopy(e["b"]), factors=factors,
                          ordered_music=ordered, alignment=alignment, temporal_comparison=sequence_comparison(ordered),
                          structure_context=copy.deepcopy(e["structure_context"])))
    payload = dict(format="compact_evidence_v2", evidence=items, backend=backend,
                   retrieval=dict(threshold=comparison.get("threshold"),matrix_maxima=comparison.get("matrix_maxima",{}),
                                  total_window_pairs=comparison.get("total_window_pairs"),
                                  status="candidates_observed" if items else "no_candidates_over_threshold" if any(v is not None for v in comparison.get("matrix_maxima",{}).values()) else "no_supported_features"),
                   score_definition=SCORE_DEFINITION, selection=dict(method="bilateral_nms", overlap=.6, limit=limit),
                   validation_scope="predicted/derived evidence; no human-validated analyzer accuracy",
                   limitations=["structure_labels_are_context_only", "unavailable_is_not_low_similarity",
                                "window_alignment_is_not_note_alignment"])
    return payload


def calibrate_scores(manifest_path, backend, out, split="validation", min_per_class=2):
    """Validation-only balanced-accuracy thresholds, NOT probabilities."""
    from .audio import load_audio,Window
    from .data import save_json
    from .train import audit_manifest,alignment_specs
    if split != "validation":
        raise ValueError("Score calibration is allowed only on validation")
    manifest = json.loads(Path(manifest_path).read_text())
    audit_manifest(manifest)
    cache, rows = {}, {f: [] for f in backend.supported}
    for pair in manifest["pairs"]:
        if pair["split"] != split:
            continue
        for astart,bstart in alignment_specs(pair):
            embeddings=[]
            for side,start in (("a",astart),("b",bstart)):
                path=pair[side]["audio"];key=(path,start)
                if path not in cache:cache[path]=load_audio(path)
                if key not in cache:
                    audio=cache[path];begin=round(start*24000);samples=audio.samples[begin:begin+240000]
                    window=Window(0,start,start+len(samples)/24000,np.pad(samples,(0,240000-len(samples))),len(samples))
                    emb,_=backend.encode(window)
                    cache[key]={f:np.asarray(v)/max(float(np.linalg.norm(v)),1e-8) for f,v in emb.items()}
                embeddings.append(cache[key])
            for factor in rows:
                label=pair["targets"].get("pitch_intervals" if factor=="melody" else factor)
                if label is not None and all(factor in e for e in embeddings):
                    rows[factor].append((int(label),float(np.clip(embeddings[0][factor]@embeddings[1][factor],-1,1))))
    factors = {}
    for factor, values in rows.items():
        pos, neg = sum(y == 1 for y, _ in values), sum(y == 0 for y, _ in values)
        info = dict(status="uncalibrated", method="balanced_accuracy_threshold", n=len(values), positives=pos, negatives=neg,
                    probability_calibration=False,score_unit="aligned_window_pair_cosine",
                    label_scope="controlled_pair_factor_labels_applied_to_time_aligned_windows; not human labels",
                    validation_original_ids=sorted(k for k,v in manifest.get("split_originals", {}).items() if v=="validation"))
        if min(pos, neg) >= min_per_class:
            candidates = sorted({-1., 1.000001, *[v for _, v in values]})
            def balanced(t):
                return .5 * (sum(s >= t for y, s in values if y == 1)/pos + sum(s < t for y, s in values if y == 0)/neg)
            best = max(candidates, key=lambda t: (balanced(t), t))
            info.update(status="fitted_validation_small_sample" if len(values) < 100 else "fitted_validation",
                        threshold=best, selection_balanced_accuracy=balanced(best))
        else:
            info["reason"] = "insufficient_samples_per_class"
        factors[factor] = info
    result = dict(schema_version=1, split=split, method="validation_threshold_not_probability", factors=factors,
                  backend_sha256=digest(backend.metadata()), backend=backend.metadata(), manifest_sha256=file_digest(manifest_path))
    result["checkpoint_sha256"]=file_digest(backend.checkpoint) if hasattr(backend,"checkpoint") else None
    save_json(out, result)
    return result

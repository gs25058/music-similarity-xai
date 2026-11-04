import copy
from pathlib import Path

import numpy as np
import soundfile as sf

from .data import materialize_record, render, save_json, split_originals, transform
from .features import FACTORS


def compose(index, duration=10.0):
    rng = np.random.default_rng(1000+index)
    beat = float(rng.choice([.4, .5, .625]))
    melody, accompaniment, bass, chords = [], [], [], []
    scale = np.array([0, 2, 4, 5, 7, 9, 11])
    tonic = index % 12
    pattern = rng.choice(scale, 12) + 60 + tonic
    intervals = rng.choice([.5, 1, 1.5], 12)
    time, k = 0., 0
    while time < duration:
        length = float(intervals[k % 12]*beat)
        melody.append(dict(start=time, end=min(duration, time+length*.85), pitch=int(pattern[k%12]), velocity=95))
        time += length
        k += 1
    progression = rng.choice([0, 2, 5, 7, 9], 5)
    for start in np.arange(0, duration, 2):
        root = int((tonic+progression[int(start/2) % 5]) % 12)
        quality = int(rng.integers(0, 2))
        chords.append(dict(start=float(start), end=min(duration, float(start+2)), root=root, quality=quality, source="synthetic_score"))
        for pc in (root, root+(3 if quality else 4), root+7):
            accompaniment.append(dict(start=float(start), end=min(duration, float(start+1.8)), pitch=48+pc%12, velocity=75))
        for off in (0., .5, 1., 1.5):
            if start+off < duration:
                bass.append(dict(start=float(start+off), end=min(duration, float(start+off+.35)), pitch=36+root, velocity=85))
    return dict(original_id=f"synth_{index:03}", source="synthetic", duration=duration,
                melody=melody, accompaniment=accompaniment, bass=bass, bridge=[],
                beats=np.arange(0, duration, beat).tolist(), chords=chords,
                structure=[], accompaniment_source="synthetic_explicit_bass_score",
                annotation_provenance="procedural_exact_score")


def hard_negative(record):
    r = copy.deepcopy(record)
    # Retain timing, chords and bass exactly; alter the melody interval sequence.
    for i, note in enumerate(r["melody"]):
        note["pitch"] = 60 + ((i*5 + (i%3)*2 + 1) % 17)
    r["transform"] = dict(kind="same_chords_different_melody")
    return r


def pair_entry(a, b, split, kind, targets, original_ids, truth=None):
    return dict(a=a, b=b, split=split, kind=kind, original_ids=original_ids,
                targets=dict(zip(FACTORS, targets)), truth=truth)


def prepare_synthetic(out, originals=12, seed=42):
    out = Path(out)
    split = split_originals([f"synth_{i:03}" for i in range(originals)], seed)
    records, pairs = {}, []
    for index in range(originals):
        r = compose(index)
        sid = r["original_id"]
        r["split"] = split[sid]
        base = materialize_record(r, out/"audio", sid+"_base")
        records[sid] = (r, base)
        variants = [
            ("transpose", transform(r, transpose=3), [1, 1, None, 1]),
            ("tempo", transform(r, tempo=1.15), [1, 1, 1, 1]),
            ("instrument", transform(r, instrument=True), [1, 1, 1, 1]),
            ("hard_negative", hard_negative(r), [0, 1, 1, 1]),
        ]
        for kind, variant, targets in variants:
            item = materialize_record(variant, out/"audio", sid+"_"+kind)
            pairs.append(pair_entry(base, item, split[sid], kind, targets, [sid]))
    for sid, (r, base) in records.items():
        other = next((value for key, value in records.items() if key != sid and split[key] == split[sid]), None)
        if other:
            pairs.append(pair_entry(base, other[1], split[sid], "different_original", [0, 0, 0, 0], [sid, other[0]["original_id"]]))
    # Long audio insertion cases are generated within an already assigned split.
    insertion_cases = []
    for offset, split_name in enumerate(("train", "validation", "test")):
        ids = [sid for sid in records if split[sid] == split_name]
        if not ids:
            continue
        source, base = records[ids[0]]
        background = records[ids[-1]][0]
        motif = render(source)
        # Both full songs contain multiple contexts. Only the inserted pattern is shared.
        a = np.r_[render(hard_negative(source)), motif, render(background)[:24000*7]]
        b = np.r_[render(hard_negative(background)), render(background)[:24000*5], motif, render(hard_negative(background))[:24000*9]]
        demo = out/"insertion"/split_name
        demo.mkdir(parents=True, exist_ok=True)
        sf.write(demo/"song_a.wav", a, 24000)
        sf.write(demo/"song_b.wav", b, 24000)
        truth = dict(a=dict(start=10., end=20.), b=dict(start=15., end=25.),
                     factors={factor: 1 for factor in FACTORS})
        case = dict(a=str((demo/"song_a.wav").resolve()), b=str((demo/"song_b.wav").resolve()),
                    split=split_name, kind="insertion", original_ids=ids[:1]+ids[-1:], truth=truth)
        insertion_cases.append(case)
    manifest = dict(schema_version=1, seed=seed, split_originals=split, factors=list(FACTORS),
                    pair_label_semantics="1=controlled_same_pattern, 0=controlled_different, null=uncertain; tempo/transpose invariance per factor",
                    renderer="deterministic_oscillator_24000Hz", pairs=pairs, insertion_cases=insertion_cases)
    save_json(out/"manifest.json", manifest)
    return manifest

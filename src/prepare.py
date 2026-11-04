"""Prepare a bounded or full real-MIDI corpus after original-level splitting."""
import copy
from pathlib import Path

from .data import chord_classes, load_pop909, load_slakh, materialize_record, save_json, split_originals, transform
from .features import FACTORS
from .synthetic import hard_negative, pair_entry


def crop_record(record, start, end):
    r = copy.deepcopy(record)
    r["duration"] = end-start
    for track in ("melody", "accompaniment", "bridge", "bass"):
        r[track] = [{**n, "start": n["start"]-start, "end": min(n["end"], end)-start}
                    for n in record.get(track, []) if n["end"] > start and n["start"] < end]
    r["beats"] = [b-start for b in record.get("beats", []) if start <= b < end]
    r["chords"] = [{**c, "start": max(c["start"], start)-start, "end": min(c["end"], end)-start}
                    for c in record.get("chords", []) if c["end"] > start and c["start"] < end]
    r["structure"] = [{**s, "start": max(s["start"], start)-start, "end": min(s["end"], end)-start}
                       for s in record.get("structure", []) if s["end"] > start and s["start"] < end]
    r["original_interval"] = dict(start=start, end=end)
    return r


def controlled_negative(record, factor):
    """Other factor targets are masked: these edits may have coupled effects."""
    import math
    r = copy.deepcopy(record)
    if factor == "rhythm":
        def warp(t):
            return min(r["duration"], max(0., t+.12*math.sin(2*math.pi*t/3)))
        for key in ("melody", "accompaniment", "bridge", "bass"):
            for n in r.get(key, []):
                n["start"], n["end"] = warp(n["start"]), warp(n["end"])
            r[key] = [n for n in r.get(key, []) if n["end"] > n["start"]]
        r["beats"] = [warp(b) for b in r.get("beats", [])]
        for chord in r.get("chords", []):
            chord["start"], chord["end"] = warp(chord["start"]), warp(chord["end"])
    elif factor == "harmony":
        for n in r.get("accompaniment", []):
            shift = (1, 4, -2)[int(n["start"]//2) % 3]
            n["pitch"] = min(127, max(0, n["pitch"]+shift))
        for chord in r.get("chords", []):
            parsed = chord_classes(chord)
            if parsed and parsed[0] < 12:
                chord["root"] = (parsed[0]+(1, 4, -2)[int(chord["start"]//2) % 3]) % 12
                chord["quality"] = parsed[1]
    elif factor == "accompaniment_bass":
        key = "bass" if r.get("bass") else "accompaniment"
        for index, n in enumerate(r.get(key, [])):
            n["pitch"] = min(127, max(0, n["pitch"]+(7 if index % 2 else -4)))
    else:
        raise ValueError(factor)
    r["transform"] = dict(kind="controlled_negative", factor=factor, other_factors="masked")
    return r


def prepare_real(out, pop909=None, slakh=None, slakh_redux=None, structure_root=None,
                 max_originals=0, max_segments=0, seed=42):
    records = []
    if pop909:
        pop = load_pop909(pop909, structure_root)
        membership = split_originals([r["original_id"] for r in pop], seed)
        for r in pop:
            r["split"] = membership[r["original_id"]]
        records.extend(pop)
    if slakh:
        if not slakh_redux:
            raise ValueError("--slakh-redux is required; legacy Slakh splits may leak duplicate MIDI")
        records.extend(load_slakh(slakh, slakh_redux))
    if not records:
        raise ValueError("No MIDI records found")
    membership = {r["original_id"]: r["split"] for r in records}
    # Optional sample limit is applied after the full corpus split is established.
    if max_originals:
        records = records[:max_originals]
    out = Path(out)
    pairs = []
    for record in records:
        starts = list(range(0, max(1, int(record["duration"])-4), 10))
        if max_segments:
            starts = starts[:max_segments]
        for index, start in enumerate(starts):
            crop = crop_record(record, start, min(record["duration"], start+10))
            if crop["duration"] < .1:
                continue
            name = record["original_id"]+f"_{index:03}"
            base = materialize_record(crop, out/"audio", name)
            has_melody = bool(crop.get("melody"))
            has_rhythm = any(crop.get(k) for k in ("melody", "accompaniment", "bass"))
            has_harmony = bool(crop.get("accompaniment") or crop.get("chords"))
            has_bass = bool(crop.get("bass") or (crop.get("accompaniment") and
                            crop.get("accompaniment_source") == "piano_accompaniment_low_register_derived"))
            targets = [1 if v else None for v in (has_melody, has_rhythm, has_harmony, has_bass)]
            variants = [("instrument", transform(crop, instrument=True), targets),
                        ("tempo", transform(crop, tempo=1.15), targets),
                        ("transpose", transform(crop, transpose=3), [targets[0], targets[1], None, targets[3]])]
            if has_melody:
                # Changing melody also changes onset-combined chroma; mask harmony.
                variants.append(("hard_negative", hard_negative(crop), [0, targets[1], None, targets[3]]))
            for factor_index, factor in enumerate(FACTORS):
                if factor_index and targets[factor_index] is not None:
                    if factor == "harmony" and len({int(n["start"]//2) % 3 for n in crop.get("accompaniment", [])}) < 2:
                        continue  # a single global shift is ambiguous for harmony
                    if factor == "accompaniment_bass" and len(crop.get("bass") or crop.get("accompaniment", [])) < 3:
                        continue
                    variant = controlled_negative(crop, factor)
                    if all(variant.get(k) == crop.get(k) for k in ("melody", "accompaniment", "bass", "beats", "chords")):
                        continue
                    labels = [None]*len(FACTORS)
                    labels[factor_index] = 0
                    variants.append((factor+"_negative", variant, labels))
            for kind, variant, labels in variants:
                item = materialize_record(variant, out/"audio", name+"_"+kind)
                pairs.append(pair_entry(base, item, record["split"], kind, labels, [record["original_id"]]))
    manifest = dict(schema_version=1, seed=seed, split_originals=membership, pairs=pairs, insertion_cases=[],
                    factors=list(FACTORS), provenance=dict(pop909=pop909, slakh=slakh, slakh_redux=slakh_redux),
                    note="Real MIDI rendered with fixed oscillator for controlled augmentation; Slakh original mix/stem paths remain in annotations. Slakh melody is unavailable unless separately annotated.")
    save_json(out/"manifest.json", manifest)
    return manifest

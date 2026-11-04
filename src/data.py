"""MIDI/event supervision. Audio is always the inference input."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
import yaml

from .features import FPS, MIDI_MIN, MIDI_MAX


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def split_originals(ids, seed=42):
    """Assign original IDs before creating any windows, variants, or pairs."""
    ids = sorted(set(ids))
    if len(ids) < 3:
        raise ValueError("Need at least 3 original groups for train/validation/test")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = max(1, round(.2*len(ids)))
    n_val = max(1, round(.2*len(ids)))
    return {sid: ("test" if i < n_test else "validation" if i < n_test+n_val else "train")
            for i, sid in enumerate(ids)}


def note_dict(note):
    return dict(start=float(note.start), end=float(note.end), pitch=int(note.pitch), velocity=int(note.velocity))


def load_pop909(root, structure_root=None):
    """POP909 tracks are not verse/chorus labels and BRIDGE is not a section."""
    root = Path(root)
    records = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        path = folder / (folder.name + ".mid")
        if not path.exists():
            continue
        midi = pretty_midi.PrettyMIDI(str(path))
        tracks = {"MELODY": [], "PIANO": [], "BRIDGE": []}
        for inst in midi.instruments:
            if inst.name.upper() in tracks:
                tracks[inst.name.upper()].extend(note_dict(n) for n in inst.notes)
        beats = []
        beat_file = folder / "beat_midi.txt"
        if beat_file.exists():
            beats = np.atleast_2d(np.loadtxt(beat_file))[:, 0].astype(float).tolist()
        chords = []
        chord_file = folder / "chord_midi.txt"
        if chord_file.exists():
            for line in chord_file.read_text().splitlines():
                fields = line.split()
                if len(fields) >= 3:
                    chords.append(dict(start=float(fields[0]), end=float(fields[1]), symbol=fields[2],
                                       source="pop909_algorithm_annotation"))
        structure = []
        if structure_root:
            structure_file = Path(structure_root) / (folder.name + ".json")
            if structure_file.exists():
                structure = json.loads(structure_file.read_text())
        records.append(dict(original_id="pop909_"+folder.name, source="pop909", duration=midi.get_end_time(),
                            melody=tracks["MELODY"], accompaniment=tracks["PIANO"], bridge=tracks["BRIDGE"],
                            beats=beats, chords=chords, structure=structure,
                            accompaniment_source="piano_accompaniment_low_register_derived",
                            annotation_provenance="pop909_midi_and_algorithm_annotations", midi_path=str(path)))
    return records


def load_slakh(root, redux_json):
    """Require official redux split; group by UUID and reject cross-split duplicates."""
    root = Path(root)
    splits = json.loads(Path(redux_json).read_text())
    membership = {}
    if splits and all(key.startswith("Track") and isinstance(value, dict) for key, value in splits.items()):
        # Official slakh-utils redux.json is an action map, not split -> IDs.
        for tid, action in splits.items():
            if action.get("action") == "omit":
                continue
            if action.get("action") not in (None, "move"):
                raise ValueError(f"Unknown Slakh split action for {tid}")
            number = int(tid.removeprefix("Track"))
            original_split = "train" if number <= 1500 else "validation" if number <= 1875 else "test"
            membership[tid] = action["destination_split"] if action.get("action") == "move" else original_split
    else:
        # Also accept an explicit exported split membership map for local subsets.
        for split, ids in splits.items():
            canonical = {"valid": "validation", "val": "validation"}.get(split, split)
            if canonical not in ("train", "validation", "test"):
                continue
            for tid in ids:
                if tid in membership:
                    raise ValueError(f"Duplicate redux member: {tid}")
                membership[tid] = canonical
    if not membership:
        raise ValueError("Empty or unsupported Slakh redux split file")
    records, seen = [], {}
    for path in sorted(root.rglob("metadata.yaml")):
        tid = path.parent.name
        if tid not in membership:
            continue
        meta = yaml.safe_load(path.read_text())
        uuid = str(meta.get("UUID") or hashlib.sha256(str(meta.get("lmd_midi_dir", tid)).encode()).hexdigest())
        split = membership[tid]
        if uuid in seen:
            if seen[uuid] != split:
                raise ValueError(f"MIDI duplicate crosses Slakh splits: {uuid}")
            continue
        seen[uuid] = split
        bass, accompaniment, stem_audio, availability_warnings = [], [], [], []
        duration = 0.0
        for stem, details in meta.get("stems", {}).items():
            midi_path = path.parent / meta.get("midi_dir", "MIDI") / (stem+".mid")
            audio_path = next((path.parent/meta.get("audio_dir", "stems")/(stem+ext) for ext in (".flac", ".wav")
                               if (path.parent/meta.get("audio_dir", "stems")/(stem+ext)).exists()), None)
            notes = []
            if bool(details.get("midi_saved")) != midi_path.exists():
                availability_warnings.append(dict(stem=stem, field="midi_saved", metadata=details.get("midi_saved"), file_exists=midi_path.exists()))
            if midi_path.exists():
                midi = pretty_midi.PrettyMIDI(str(midi_path))
                notes = [note_dict(n) for i in midi.instruments if not i.is_drum for n in i.notes]
                duration = max(duration, midi.get_end_time())
            is_bass = not details.get("is_drum", False) and 32 <= int(details.get("program_num", -1)) <= 39
            if is_bass:
                bass.extend(notes)
            elif not details.get("is_drum", False):
                accompaniment.extend(notes)
            if bool(details.get("audio_rendered")) != (audio_path is not None):
                availability_warnings.append(dict(stem=stem, field="audio_rendered", metadata=details.get("audio_rendered"), file_exists=audio_path is not None))
            if audio_path is not None:
                stem_audio.append(dict(stem=stem, path=str(audio_path), is_bass=is_bass,
                                       program_num=details.get("program_num")))
        mix = next((path.parent/n for n in ("mix.flac", "mix.wav") if (path.parent/n).exists()), None)
        records.append(dict(original_id="slakh_"+uuid, track_id=tid, source="slakh", split=split,
                            duration=duration, melody=[], accompaniment=accompaniment, bass=bass, bridge=[],
                            beats=[], chords=[], structure=[], stem_audio=stem_audio,
                            availability_warnings=availability_warnings,
                            audio_path=str(mix) if mix else None, accompaniment_source="actual_bass_midi" if bass else "unavailable",
                            annotation_provenance="slakh_metadata_and_stem_midi"))
    return records


def load_stem_audio(record, bass_only=False):
    """Lazily decode selected Slakh stems into 24 kHz mono Audio objects."""
    from .audio import load_audio
    return [{**stem, "audio": load_audio(stem["path"])} for stem in record.get("stem_audio", [])
            if not bass_only or stem["is_bass"]]


def transform(record, transpose=0, tempo=1.0, instrument=False):
    r = copy.deepcopy(record)
    r["duration"] /= tempo
    for track in ("melody", "accompaniment", "bridge", "bass"):
        for n in r.get(track, []):
            n["pitch"] += transpose
            n["start"] /= tempo
            n["end"] /= tempo
    r["beats"] = [b/tempo for b in r.get("beats", [])]
    for chord in r.get("chords", []):
        chord["start"] /= tempo
        chord["end"] /= tempo
        if "root" in chord:
            chord["root"] = (chord["root"]+transpose) % 12
        else:
            # Symbol transposition is intentionally not guessed.
            chord["masked"] = bool(transpose)
    for section in r.get("structure", []):
        section["start"] /= tempo
        section["end"] /= tempo
    r["render_instrument"] = "harmonic" if instrument else r.get("render_instrument", "sine")
    r["transform"] = dict(transpose=transpose, tempo=tempo, instrument=instrument)
    return r


def render(record, sr=24000):
    """Reproducible oscillator renderer, no external soundfont or audio files."""
    x = np.zeros(round(record["duration"]*sr), dtype=np.float32)
    for track, gain in (("melody", .22), ("accompaniment", .055), ("bridge", .04), ("bass", .12)):
        for note in record.get(track, []):
            start, end = max(0, round(note["start"]*sr)), min(len(x), round(note["end"]*sr))
            if end <= start:
                continue
            t = np.arange(end-start)/sr
            frequency = 440*2**((note["pitch"]-69)/12)
            y = np.sin(2*np.pi*frequency*t)
            if record.get("render_instrument") == "harmonic":
                y = (y + .4*np.sin(4*np.pi*frequency*t) + .2*np.sin(6*np.pi*frequency*t))/1.6
            envelope = np.minimum(t/.012, 1)*np.minimum((len(t)/sr-t)/.04, 1)
            x[start:end] += (gain * note.get("velocity", 90)/100 * y * envelope).astype(np.float32)
    return np.clip(x, -.98, .98)


def chord_classes(chord):
    if chord.get("masked"):
        return None
    if "root" in chord:
        return int(chord["root"]), int(chord["quality"])
    symbol = chord.get("symbol", "")
    if symbol == "N":
        return 12, 4
    roots = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    parts = symbol.split(":")
    root = parts[0]
    if not root or root[0] not in roots:
        return None
    accidental = root[1:]
    if accidental not in ("", "b", "#"):
        return None
    pc = (roots[root[0]] + {"": 0, "b": -1, "#": 1}[accidental]) % 12
    quality = parts[1].split("/")[0] if len(parts)>1 else "maj"
    mapping = {"maj": 0, "min": 1, "dim": 2, "aug": 3}
    return (pc, mapping[quality]) if quality in mapping else None


def frame_labels(record, start, total_frames, valid):
    times = start + np.arange(total_frames)/FPS
    labels = {}
    for task in ("melody_pitch", "melody_onset", "beat", "onset", "chroma", "chord_root", "chord_quality", "bass_pitch", "bass_onset"):
        labels[task] = np.zeros((total_frames, 12) if task == "chroma" else total_frames, dtype=np.float32)
        labels[task+"_mask"] = np.zeros(total_frames, dtype=bool)
    tracks = [("melody", "melody")]
    bass_key = "bass" if record.get("bass") else (
        "accompaniment" if record.get("accompaniment_source") == "piano_accompaniment_low_register_derived" else None)
    if bass_key:
        tracks.append((bass_key, "bass"))
    for key, prefix in tracks:
        notes = record.get(key, [])
        if not notes:
            continue
        labels[prefix+"_pitch"][:] = 84
        for task in (prefix+"_pitch", prefix+"_onset"):
            labels[task+"_mask"][:] = valid
        for i, t in enumerate(times):
            active = [n["pitch"] for n in notes if n["start"] <= t < n["end"]]
            if active:
                pitch = max(active) if prefix == "melody" else min(active)
                if MIDI_MIN <= pitch <= MIDI_MAX:
                    labels[prefix+"_pitch"][i] = pitch-MIDI_MIN
                else:
                    labels[prefix+"_pitch_mask"][i] = False
        for n in notes:
            idx = round((n["start"]-start)*FPS)
            if 0 <= idx < total_frames:
                labels[prefix+"_onset"][idx] = 1
    all_notes = [n for key in ("melody", "accompaniment", "bridge", "bass") for n in record.get(key, [])]
    if all_notes:
        labels["onset_mask"][:] = valid
        for n in all_notes:
            idx = round((n["start"]-start)*FPS)
            if 0 <= idx < total_frames:
                labels["onset"][idx] = 1
    harmonic_notes = record.get("accompaniment", [])
    if harmonic_notes:
        labels["chroma_mask"][:] = valid
        for n in harmonic_notes:
            labels["chroma"][(times >= n["start"]) & (times < n["end"]), n["pitch"] % 12] = 1
    for beat in record.get("beats", []):
        idx = round((beat-start)*FPS)
        if 0 <= idx < total_frames:
            labels["beat"][idx] = 1
    if record.get("beats"):
        labels["beat_mask"][:] = valid
    for chord in record.get("chords", []):
        parsed = chord_classes(chord)
        if parsed:
            mask = (times >= chord["start"]) & (times < chord["end"]) & valid
            for task, value in zip(("chord_root", "chord_quality"), parsed):
                labels[task][mask] = value
                labels[task+"_mask"][mask] = True
            root, quality = parsed
            if root < 12 and quality < 4:
                labels["chroma"][mask] = 0
                intervals = ((0, 4, 7), (0, 3, 7), (0, 3, 6), (0, 4, 8))[quality]
                for pc in intervals:
                    labels["chroma"][mask, (root+pc) % 12] = 1
                labels["chroma_mask"][mask] = True
    return labels


def materialize_record(record, root, name):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sf.write(root/(name+".wav"), render(record), 24000)
    save_json(root/(name+".json"), record)
    return dict(audio=str((root/(name+".wav")).resolve()), annotation=str((root/(name+".json")).resolve()))

import copy
import json

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
import torch
import yaml

from src.audio import Audio, load_audio, windows
from src.backends import DSPBackend
from src.compare import analyze_song, compare_songs
from src.data import frame_labels, load_pop909, load_slakh
from src.features import frame_features
from src.judge import judge_pair, validate_judgment
from src.model import FactorModel, masked_prediction_loss, pair_loss
from src.synthetic import compose, prepare_synthetic
from src.train import audit_manifest


def test_resampling_and_final_window_time(tmp_path):
    path = tmp_path/"stereo.wav"
    sf.write(path, np.ones((44100*21, 2))*.1, 44100)
    audio = load_audio(path)
    assert audio.duration == pytest.approx(21)
    chunks = list(windows(audio))
    assert [w.start for w in chunks] == [0, 5, 10, 15]
    assert chunks[-1].end == 21
    assert chunks[-1].metadata()["padding_samples"] == 4*24000
    assert len(chunks[-1].samples) == 240000
    assert len(list(windows(Audio("exact", np.ones(240000))))) == 1


def test_masks_missing_supervision_and_padding():
    record = compose(0)
    record["melody"] = []
    record["beats"] = []
    record["chords"] = []
    record["bass"] = []
    record["source"] = "slakh"
    record["accompaniment_source"] = "unavailable"
    valid = np.r_[np.ones(30, bool), np.zeros(10, bool)]
    labels = frame_labels(record, 0, 40, valid)
    assert not labels["melody_pitch_mask"].any()
    assert not labels["bass_pitch_mask"].any()  # guitar is not relabeled as bass
    assert not labels["beat_mask"].any()
    assert not labels["chord_root_mask"].any()
    model = FactorModel()
    output = model(torch.randn(1, 40, 86), torch.tensor(valid[None]))
    batch = {k: torch.tensor(v[None]) for k, v in labels.items()}
    loss = masked_prediction_loss(output, batch, torch.tensor(valid[None]))
    changed = {k: v.clone() for k, v in batch.items()}
    changed["chord_root"][:] = 999  # unobserved targets must never enter CE
    changed["chroma"][:, 30:] = 999
    assert torch.equal(loss, masked_prediction_loss(output, changed, torch.tensor(valid[None])))
    no_targets = {k: (torch.zeros_like(v) if k.endswith("_mask") else v) for k, v in batch.items()}
    zero = masked_prediction_loss(output, no_targets, torch.tensor(valid[None]))
    assert zero.item() == 0
    zero.backward()


def test_pair_loss_masks_unknown_factors():
    model = FactorModel()
    a = model(torch.randn(2, 20, 86), torch.ones(2, 20, dtype=torch.bool))
    b = model(torch.randn(2, 20, 86), torch.ones(2, 20, dtype=torch.bool))
    assert pair_loss(a, b, torch.ones(2, 4), torch.zeros(2, 4, dtype=torch.bool)).item() == 0


def test_no_original_id_leakage(tmp_path):
    manifest = prepare_synthetic(tmp_path, originals=6)
    audit_manifest(manifest)
    broken = copy.deepcopy(manifest)
    broken["pairs"][0]["split"] = "test" if broken["pairs"][0]["split"] != "test" else "train"
    with pytest.raises(ValueError, match="leakage"):
        audit_manifest(broken)


def test_silence_and_structure_context_do_not_create_similarity(tmp_path):
    path = tmp_path/"silence.wav"
    sf.write(path, np.zeros(24000), 24000)
    backend = DSPBackend()
    song = analyze_song(path, backend)
    comparison = compare_songs(song, song, backend)
    assert comparison["evidence"] == []
    assert judge_pair(comparison)["selected"]["judgment"] == "insufficient_evidence"
    song["structure"] = dict(status="executed", segments=[dict(start=0, end=1, label="chorus")])
    assert compare_songs(song, song, backend)["evidence"] == []


def test_judge_rejects_hallucinated_ids_and_times():
    evidence = [dict(evidence_id="E001")]
    valid = dict(judgment="review", evidence_ids=["E001"], explanation="E001의 음악 근거를 검토하세요.")
    assert validate_judgment(valid, evidence) == valid
    for updates in (dict(evidence_ids=["E999"]), dict(explanation="E999가 유사합니다."),
                    dict(explanation="10초에서 유사합니다."), dict(explanation="English only"),
                    dict(evidence_ids=[]), dict(judgment="plagiarism"), dict(explanation="유사도는 0.99입니다.")):
        with pytest.raises(ValueError):
            validate_judgment({**valid, **updates}, evidence)
    context = [{**evidence[0], "structure_context": {"a": [{"label": "verse"}], "b": []}}]
    with pytest.raises(ValueError, match="observed"):
        validate_judgment({**valid, "explanation": "구조 정보를 관측하지 못했습니다."}, context)


def test_pop909_tracks_are_not_section_labels(tmp_path):
    folder = tmp_path/"001"
    folder.mkdir()
    midi = pretty_midi.PrettyMIDI()
    for name, pitch in (("MELODY", 72), ("PIANO", 48), ("BRIDGE", 67)):
        instrument = pretty_midi.Instrument(0, name=name)
        instrument.notes.append(pretty_midi.Note(100, pitch, 0, 1))
        midi.instruments.append(instrument)
    midi.write(str(folder/"001.mid"))
    records = load_pop909(tmp_path)
    assert len(records[0]["bridge"]) == 1
    assert records[0]["structure"] == []
    assert records[0]["accompaniment_source"] == "piano_accompaniment_low_register_derived"


def test_slakh_bass_metadata_and_duplicate_split(tmp_path):
    for tid in ("Track00001", "Track00002"):
        folder = tmp_path/tid
        (folder/"MIDI").mkdir(parents=True)
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(33)
        instrument.notes.append(pretty_midi.Note(100, 40, 0, 1))
        midi.instruments.append(instrument)
        midi.write(str(folder/"MIDI/S00.mid"))
        (folder/"metadata.yaml").write_text(yaml.safe_dump(dict(UUID="same_uuid", stems={"S00": dict(program_num=33, is_drum=False, midi_saved=True, audio_rendered=False)})))
    redux = tmp_path/"redux.json"
    redux.write_text(json.dumps(dict(train=["Track00001"], test=["Track00002"])))
    with pytest.raises(ValueError, match="crosses"):
        load_slakh(tmp_path, redux)
    redux.write_text(json.dumps(dict(train=["Track00001"])))
    record = load_slakh(tmp_path, redux)[0]
    assert record["bass"][0]["pitch"] == 40
    assert record["melody"] == []
    assert record["accompaniment_source"] == "actual_bass_midi"
    # Official action-map schema must omit the second track without moving files.
    redux.write_text(json.dumps({"Track00001": {"action": None},
                                "Track00002": {"action": "omit", "source_split": "train"}}))
    assert len(load_slakh(tmp_path, redux)) == 1
    folder = tmp_path/"Track00001"
    (folder/"stems").mkdir()
    sf.write(folder/"stems/S00.wav", np.ones(1600)*.1, 16000)
    metadata = yaml.safe_load((folder/"metadata.yaml").read_text())
    metadata["stems"]["S00"]["midi_saved"] = False
    (folder/"metadata.yaml").write_text(yaml.safe_dump(metadata))
    record = load_slakh(tmp_path, redux)[0]
    assert record["bass"] and len(record["availability_warnings"]) == 2
    from src.data import load_stem_audio
    decoded = load_stem_audio(record, bass_only=True)[0]["audio"]
    assert decoded.sample_rate == 24000 and decoded.duration == pytest.approx(.1)


def test_temporal_chain_preserves_precise_peak_window():
    def window(index, embedding):
        return dict(index=index, start=index*5., end=index*5.+10,
                    actual_duration=10., padded_duration=10.,
                    embeddings={"rhythm": np.asarray(embedding)}, music_information={})
    backend = DSPBackend()
    backend.supported = ("rhythm",)
    song = dict(duration=15., windows=[window(0, [1., 0.]), window(1, [0., 1.])],
                structure=dict(status="unavailable", segments=[]))
    result = compare_songs(song, song, backend)
    assert any(e["candidate_kind"] == "monotone_chain" and e["a"]["end"] == 15 for e in result["evidence"])
    assert result["evidence"][0]["candidate_kind"] == "peak_window"
    assert result["evidence"][0]["a"]["end"]-result["evidence"][0]["a"]["start"] == 10

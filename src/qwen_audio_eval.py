"""A separate waveform intervention; never pooled with frozen-evidence tests."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import load_audio
from .compare import analyze_song,compare_songs,public_song
from .data import save_json
from .evidence import file_digest
from .judge import judge_pair
from .evaluate import interval_iou


def zero_audio_region(samples,start,end,sample_rate=24000):
    if not 0<=start<end<=len(samples)/sample_rate:raise ValueError("Intervention outside source duration")
    result=np.array(samples,copy=True);result[round(start*sample_rate):round(end*sample_rate)]=0
    return result


def evaluate_audio_intervention(manifest,backend,out,config,qwen_python,device="cpu",structure=None,split="test"):
    m=json.loads(Path(manifest).read_text());case=next((x for x in m.get("insertion_cases",[]) if x["split"]==split),None)
    if case is None:raise ValueError("Need a known insertion interval; do not invent a global similarity gold label")
    cfg=json.loads(Path(config).read_text())
    if cfg.get("adapter"):
        meta=json.loads((Path(cfg["adapter"])/"training_meta.json").read_text())
        if set(meta.get("train_original_ids",[]))&set(case["original_ids"]):raise ValueError("Audio intervention uses LoRA training originals")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    b=load_audio(case["b"]);truth=case["truth"]["b"]
    changed=zero_audio_region(b.samples,truth["start"],truth["end"])
    path=out/"b_removed_insert.wav";sf.write(path,changed,24000,subtype="FLOAT")
    a=analyze_song(case["a"],backend,structure=structure)
    results=[]
    for name,b_path in (("original",case["b"]),("removed_insert_audio",str(path))):
        analysis_b=analyze_song(b_path,backend,structure=structure)
        comparison=compare_songs(a,analysis_b,backend)
        judgment=judge_pair(comparison,"qwen",qwen_python,device,cfg,dict(backend=backend.metadata(),songs=dict(a=a,b=analysis_b)))
        report=dict(schema_version=1,task="musical_similarity_for_further_review",backend=backend.metadata(),
                    songs=dict(a=public_song(a),b=public_song(analysis_b)),comparison=comparison,judgment=judgment)
        report_path=out/(name+".json");save_json(report_path,report)
        top=comparison["evidence"][0] if comparison["evidence"] else None
        results.append(dict(variant=name,report=str(report_path),qwen_status=judgment["execution"]["status"],
            verdict=judgment["selected"]["judgment"],audio_sha256=dict(a=file_digest(case["a"]),b=file_digest(b_path)),
            top1_original_insert_bilateral_iou=min(interval_iou(top["a"],case["truth"]["a"]),interval_iou(top["b"],case["truth"]["b"])) if top else 0.))
    result=dict(evaluation_mode="audio_reanalysis",backend=backend.metadata(),manifest_sha256=file_digest(manifest),
        source_original_ids=case["original_ids"],split=split,intervention=dict(kind="zero_known_B_insert_region",interval=truth,
            source_preserved=True,cut_boundary_transients_possible=True),results=results,
        global_verdict_gold=None,note="Other matching passages may remain; do not require a global verdict flip. All configured upstream analyzers were rerun.")
    save_json(out/"summary.json",result)
    return result

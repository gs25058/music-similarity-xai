"""Optional frozen temporal-head features -> 16 learned Qwen prefix vectors."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .evidence import digest,file_digest

FEATURE_DEFINITION="paired_frozen_head64_20hz_pad10s_v1"


class SoftConnector(nn.Module):
    def __init__(self,input_dim=262,hidden_dim=128,output_dim=1024,token_count=16,scale=.02):
        super().__init__()
        self.input_dim=input_dim;self.output_dim=output_dim;self.token_count=token_count;self.scale=scale
        self.projection=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,output_dim))

    def forward(self,features):
        if features.ndim==2:features=features.unsqueeze(0)
        if features.shape[-1]!=self.input_dim:raise ValueError("Soft feature dimensionality mismatch")
        if not features.shape[1]:
            return features.new_zeros((features.shape[0],self.token_count,self.output_dim))
        projected=self.projection(features.to(next(self.parameters()).dtype))*self.scale
        return F.adaptive_avg_pool1d(projected.transpose(1,2),self.token_count).transpose(1,2)


def prepend_soft_tokens(model,batch,labels,connector,features):
    """Text and prefix padding masks also define monotone text positions.

    No musical features means an entirely masked prefix, so ablation cases
    cannot obtain removed evidence from a learned unconditional prefix.
    """
    text=model.get_input_embeddings()(batch["input_ids"])
    prefix=connector(features).to(text.dtype)
    if prefix.shape[0]==1 and text.shape[0]>1:prefix=prefix.expand(text.shape[0],-1,-1)
    if prefix.shape[-1]!=text.shape[-1]:raise ValueError("Connector output must match actual Qwen embedding dimension")
    available=int(features.shape[-2]>0)
    mask=torch.cat([torch.full((text.shape[0],connector.token_count),available,device=text.device,dtype=batch["attention_mask"].dtype),batch["attention_mask"]],-1)
    positions=(mask.long().cumsum(-1)-1).clamp_min(0).masked_fill(mask==0,0)
    result=dict(inputs_embeds=torch.cat([prefix,text],1),attention_mask=mask,position_ids=positions)
    if "mm_token_type_ids" in batch:
        result["mm_token_type_ids"]=F.pad(batch["mm_token_type_ids"],(connector.token_count,0))
    if labels is not None:labels=F.pad(labels,(connector.token_count,0),value=-100)
    return result,labels


def swap_feature_sides(features):
    if features.shape[-1]!=262:raise ValueError("Expected paired 64-dimensional temporal heads")
    out=features.clone()
    out[...,:64]=features[...,64:128];out[...,64:128]=features[...,:64]
    out[...,128:192]=-features[...,128:192]
    out[...,-2]=features[...,-1];out[...,-1]=features[...,-2]
    return out


def load_connector(adapter,device):
    path=Path(adapter);spec=json.loads((path/"soft_config.json").read_text())
    connector=SoftConnector(**spec).to(device)
    connector.load_state_dict(torch.load(path/"connector.pt",weights_only=True,map_location=device),strict=True)
    for parameter in connector.parameters():parameter.requires_grad_(False)
    return connector.eval()


def load_feature_cache(path,payload):
    path=Path(path);meta=json.loads(path.with_suffix(".json").read_text())
    if meta["payload_sha256"]!=digest(payload):raise ValueError("Soft cache evidence fingerprint mismatch")
    if meta["preprocessing"].get("definition")!=FEATURE_DEFINITION:raise ValueError("Soft preprocessing version mismatch; rebuild cache")
    if meta["features_sha256"]!=file_digest(path):raise ValueError("Corrupt soft feature cache")
    checkpoint=Path(meta["checkpoint"]["path"])
    if checkpoint.exists() and file_digest(checkpoint)!=meta["checkpoint"]["sha256"]:
        raise ValueError("Soft cache checkpoint fingerprint mismatch")
    for info in meta["audio"].values():
        source=Path(info["path"])
        if source.exists() and file_digest(source)!=info["sha256"]:raise ValueError("Soft cache audio fingerprint mismatch")
    with np.load(path,allow_pickle=False) as data:features=torch.from_numpy(data["features"].copy())
    if features.ndim!=2 or features.shape[1]!=262 or not torch.isfinite(features).all():raise ValueError("Invalid cached feature shape/values")
    return features,meta


def prepare_soft_caches(dataset,out,checkpoint,device="cpu",samples_per_window=16):
    """Core environment only: the Qwen worker receives cache paths, not MERT."""
    from .audio import load_audio,Window
    from .features import frame_features,FACTORS
    from .model import LearnedBackend
    from .qwen_data import read_jsonl,write_jsonl,audit_source_groups
    rows=read_jsonl(dataset);audit_source_groups(rows)
    backend=LearnedBackend(checkpoint,device)
    if backend.encoder is None:raise ValueError("Soft-token experiment requires the real frozen MERT backbone")
    backend.model.eval();backend.encoder.eval()
    for p in backend.model.parameters():p.requires_grad_(False)
    for p in backend.encoder.parameters():p.requires_grad_(False)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    audio_cache={};feature_cache={};report_cache={};report_hashes={}
    checkpoint_identity=dict(path=str(checkpoint),sha256=file_digest(checkpoint))
    def audio(path):
        if path not in audio_cache:audio_cache[path]=(load_audio(path),file_digest(path))
        return audio_cache[path]
    def hidden(path,interval):
        a,_=audio(path);key=(path,interval["start"],interval["end"])
        if key not in feature_cache:
            start,end=[round(interval[k]*24000) for k in ("start","end")]
            samples=a.samples[start:end]
            if len(samples)<800:raise ValueError("Matched interval too short for MERT")
            actual_samples=len(samples)
            samples=np.pad(samples,(0,max(0,240000-actual_samples)))
            _,valid,_=frame_features(samples,actual_samples);mask=torch.tensor(valid[None],device=device)
            with torch.inference_mode():
                frames=backend.encoder(samples,actual_samples,len(valid))
                values={factor:head.temporal((frames*mask.unsqueeze(-1)).transpose(1,2)).transpose(1,2)[0,mask[0]].float().cpu().numpy()
                        for factor,head in backend.model.heads.items() if factor in backend.supported}
            feature_cache[key]=values
        return feature_cache[key]
    for row in rows:
        report_path=row["original_evidence"]["report_path"]
        if report_path not in report_cache:
            report_hashes[report_path]=file_digest(report_path)
            report_cache[report_path]=json.loads(Path(report_path).read_text())
        if report_hashes[report_path]!=row["original_evidence"]["report_sha256"]:raise ValueError("Original prediction report fingerprint changed")
        report=report_cache[report_path]
        if report["backend"].get("checkpoint")!=str(checkpoint):raise ValueError("Feature checkpoint differs from prediction report")
        paths={side:report["songs"][side]["path"] for side in ("a","b")}
        if row["variant"]=="swap_ab":paths["a"],paths["b"]=paths["b"],paths["a"]
        identity={side:dict(path=paths[side],sha256=audio(paths[side])[1]) for side in ("a","b")}
        feature_rows=[];segments=[]
        for evidence in row["input"]["evidence"]:
            for pair in evidence["alignment"].get("windows",[]) or [dict(a=evidence["a"],b=evidence["b"])]:
                supported=[f for f in FACTORS if evidence["factors"].get(f,{}).get("status")=="available"]
                if not supported:continue
                a,b=[hidden(paths[s],pair[s]) for s in ("a","b")]
                for index,factor in enumerate(FACTORS):
                    if factor not in supported:continue
                    n=min(samples_per_window,len(a[factor]),len(b[factor]))
                    ai=np.linspace(0,len(a[factor])-1,n).astype(int);bi=np.linspace(0,len(b[factor])-1,n).astype(int)
                    av,bv=a[factor][ai],b[factor][bi]
                    kind=np.zeros((n,4),np.float32);kind[:,index]=1
                    times=np.stack([(pair["a"]["start"]+ai/20)/audio(paths["a"])[0].duration,
                                    (pair["b"]["start"]+bi/20)/audio(paths["b"])[0].duration],-1)
                    values=np.concatenate([av,bv,av-bv,np.abs(av-bv),kind,times],-1).astype(np.float32)
                    feature_rows.append(values);segments.append(dict(evidence_id=evidence["evidence_id"],factor=factor,
                        a=pair["a"],b=pair["b"],rows=n))
        features=np.concatenate(feature_rows) if feature_rows else np.empty((0,262),np.float32)
        path=out/(row["case_id"]+".npz")
        np.savez_compressed(path,features=features)
        meta=dict(schema_version=1,payload_sha256=digest(row["input"]),audio=identity,checkpoint=checkpoint_identity,
            backbone=backend.meta["model_versions"]["mert95"],
            preprocessing=dict(definition=FEATURE_DEFINITION,sample_rate=24000,fps=20,window_seconds=10,samples_per_window=samples_per_window,
                source="frozen_four_heads.temporal_after_convolution",layout="A64+B64+(A-B)64+abs(A-B)64+factor_onehot4+timeA+timeB"),
            segments=segments,evidence_ids=[e["evidence_id"] for e in row["input"]["evidence"]],features_sha256=file_digest(path),
            music_models_frozen=True,available_rows=len(features))
        path.with_suffix(".json").write_text(json.dumps(meta,ensure_ascii=False,indent=2))
        row["soft_cache"]=str(path)
    output=out/"pilot_with_soft.jsonl";write_jsonl(output,rows)
    return dict(dataset=str(output),rows=len(rows),feature_windows=len(feature_cache),input_dim=262,
                backend_checkpoint=checkpoint_identity,music_models_frozen=True)

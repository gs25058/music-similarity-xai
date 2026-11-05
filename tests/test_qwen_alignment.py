import copy
import json

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from src.claims import claim_catalog,draft_response,validate_claim_response
from src.evidence import compact_evidence,representative_evidence,temporal_observations,calibrate_scores
from src.qwen_data import audit_source_groups,intervention_cases
from src.qwen_training import collate_encoded,encode_training_row
from src.soft_tokens import SoftConnector,prepend_soft_tokens,swap_feature_sides,load_feature_cache,FEATURE_DEFINITION
from src.evidence import digest,file_digest


def fixture_payload():
    seq=temporal_observations(np.arange(60)/20,pitch=np.arange(60)%4+60,bass=np.arange(60)%3+36,
                             onset=np.arange(60)%10==0,chroma=np.ones((60,12)),source="model_prediction_not_verified")
    info=dict(status="model_prediction_not_verified",source="mert95",missing=["verified_main_melody"],temporal=seq)
    e=dict(evidence_id="E001",a=dict(start=0.,end=3.),b=dict(start=5.,end=8.),combined_cosine=.95,
        factors={f:dict(status="available" if f!="phrase_structure" else "unavailable",cosine=.95 if f!="phrase_structure" else None)
                 for f in ("pitch_intervals","rhythm","harmony","accompaniment_bass","phrase_structure")},
        window_pairs=[dict(a=0,b=0)],alignment=[dict(a=dict(start=0.,end=3.),b=dict(start=5.,end=8.),
                                                   factors=dict(rhythm=.95))],
        music_information=dict(a=[copy.deepcopy(info)],b=[copy.deepcopy(info)]),structure_context=dict(a=[dict(label="chorus")],b=[dict(label="chorus")]))
    return compact_evidence(dict(evidence=[e]),dict(backend="ours",checkpoint="fixture.pt"))


def test_temporal_order_relative_intervals_and_missing_beats():
    t=temporal_observations(np.arange(8)*.5,pitch=[60,60,62,62,61,61,64,64],onset=np.array([0,1,0,1,0,1,0,0]),max_bins=4)
    assert [v["pitch_interval"] for v in t["sequence"][1:]]==[2.,-1.,3.]
    assert t["onsets"]["beat_normalized"]["status"]=="unavailable"
    assert [v["time"] for v in t["sequence"]]==sorted(v["time"] for v in t["sequence"])
    assert t["sequence"][1]["pitch_interval_reference"]["midi"]==60
    assert temporal_observations([])["status"]=="unavailable"


def test_subsampled_intervals_keep_original_neighbor_value_and_time():
    payload=fixture_payload()
    for e in payload["evidence"]:
        for side in ("a","b"):
            for window in e["ordered_music"][side]:
                for item in window["temporal"]["sequence"]:
                    for key in ("pitch","bass"):
                        if key+"_interval" in item:
                            reference=item[key+"_interval_reference"]
                            assert item[key+"_interval"]==item[key+"_midi"]-reference["midi"]
                            assert reference["time"]<item["time"]


def test_bilateral_dedup_preserves_distinct_partner_and_identity():
    def item(i,a,b,score):return dict(evidence_id=i,a=dict(start=a,end=a+10),b=dict(start=b,end=b+10),combined_cosine=score)
    selected,aliases=representative_evidence([item("E001",0,0,.99),item("E002",1,1,.98),item("E003",1,20,.97)])
    assert [e["evidence_id"] for e in selected]==["E001","E003"]
    assert aliases=={"E001":["E002"]}


def test_claims_require_observed_predicates_and_do_not_use_section_as_similarity():
    payload=fixture_payload();response=draft_response(payload)
    assert validate_claim_response(response,payload)["valid"]
    invalid=copy.deepcopy(response);invalid["claims"][0]["evidence_ids"]=["E999"]
    assert "unknown_evidence_id" in {e["code"] for e in validate_claim_response(invalid,payload)["automatic_errors"]}
    invalid=copy.deepcopy(response);invalid["claims"][0].update(factor="phrase_structure",predicate="score_high")
    assert not validate_claim_response(invalid,payload)["valid"]
    invalid=copy.deepcopy(response);invalid["claims"][0].update(predicate="score_low")
    assert not validate_claim_response(invalid,payload)["valid"]
    assert any(c["predicate"]=="same_section_context" and c["role"]=="limits" for c in claim_catalog(payload))


def test_shared_harmony_does_not_forbid_low_verdict_with_melody_difference():
    payload=fixture_payload();payload["evidence"][0]["factors"]["pitch_intervals"]["raw_cosine"]=.2
    catalog=claim_catalog(payload)
    chosen=[next(c for c in catalog if c["factor"]==f and c["predicate"]==p) for f,p in
            [("harmony","score_high"),("pitch_intervals","score_low"),("harmony","uncalibrated_score")]]
    response=dict(claims=[dict(claim_id=f"C{i+1:03}",**c) for i,c in enumerate(chosen)],verdict="low_similarity",
        explanation="화성의 일부 유사성에 비해 음정 진행의 대응은 약합니다. 보정과 분석 정확도의 한계도 함께 검토해야 합니다.")
    assert validate_claim_response(response,payload)["valid"]
    response["claims"].pop(1)
    assert not validate_claim_response(response,payload)["valid"]


def test_numeric_direction_time_and_contradiction_are_validated():
    payload=fixture_payload();response=draft_response(payload)
    claim=next(c for c in claim_catalog(payload) if c["predicate"]=="observed_value")
    response["claims"][0]=dict(claim_id="C001",**claim)
    assert validate_claim_response(response,payload)["valid"]
    response["claims"][0]["assertion"].update(value=.99)
    assert not validate_claim_response(response,payload)["valid"]
    response=draft_response(payload)
    response["claims"][0]["assertion"]=dict(quantity="time_start",side="b",value=0,operator="eq")
    assert "numeric_direction_or_time_mismatch" in {e["code"] for e in validate_claim_response(response,payload)["automatic_errors"]}
    response=draft_response(payload);response.update(verdict="high_similarity",explanation="높은 유사성 기준이 충족되지 않습니다.")
    assert "verdict_explanation_polarity_conflict" in {e["code"] for e in validate_claim_response(response,payload)["automatic_errors"]}
    response=draft_response(payload)
    claim=next(c for c in claim_catalog(payload) if c["predicate"]=="observed_value")
    response["claims"][0]=dict(claim_id="C001",**claim)
    response["claims"][0].update(statement="원시 코사인은 0.9 이하입니다.",assertion=dict(quantity="raw_cosine",value=.9,operator="ge"))
    assert "numeric_operator_statement_conflict" in {e["code"] for e in validate_claim_response(response,payload)["automatic_errors"]}


def test_necessary_removal_vs_irrelevant_and_swaps_preserve_input():
    payload=fixture_payload();before=copy.deepcopy(payload)
    variants={kind:(p,label) for kind,p,label in intervention_cases(payload)}
    assert payload==before
    removed,label=variants["remove_unique_core"]
    assert not removed["evidence"] and label["redundant"] is False
    assert draft_response(removed)["verdict"]=="insufficient_evidence"
    assert draft_response(variants["add_irrelevant"][0])["verdict"]==draft_response(payload)["verdict"]
    assert variants["swap_ab"][0]["evidence"][0]["a"]==payload["evidence"][0]["b"]


def test_source_groups_and_originals_cannot_leak():
    rows=[dict(source_group_id="g",original_ids=["song"],split="train"),dict(source_group_id="g",original_ids=["song"],split="test")]
    with pytest.raises(ValueError,match="leakage"):audit_source_groups(rows)
    rows[1]["source_group_id"]="another"
    with pytest.raises(ValueError,match="leakage"):audit_source_groups(rows)


class CharacterTokenizer:
    eos_token_id=1
    def encode(self,text,**kwargs):return [ord(c)+2 for c in text]
    def __call__(self,text,**kwargs):return dict(input_ids=self.encode(text),offset_mapping=[(i,i+1) for i in range(len(text))])


class Processor:
    tokenizer=CharacterTokenizer()
    def apply_chat_template(self,*args,**kwargs):return "고정 근거와 assistant 시작:"


def test_output_only_unknown_fields_and_padding_are_masked():
    row=dict(case_id="fixture",input=fixture_payload(),target=draft_response(fixture_payload()),
             supervision=dict(claims=True,verdict=False,explanation=False))
    encoded=encode_training_row(Processor(),row,max_length=10000)
    assert all(v==-100 for v in encoded["labels"][:encoded["prompt_tokens"]])
    assert all(v==-100 for v,r in zip(encoded["labels"],encoded["roles"]) if r in ("verdict","explanation"))
    assert any(v!=-100 for v,r in zip(encoded["labels"],encoded["roles"]) if r=="citation")
    short=dict(input_ids=[5],labels=[-100],roles=["input"])
    batch=collate_encoded([encoded,short],0)
    assert (batch["labels"][1]==-100).all()
    assert batch["attention_mask"][1].sum()==1


def test_bpe_token_crossing_unknown_field_is_completely_masked():
    class MergedTokenizer(CharacterTokenizer):
        def __call__(self,text,**kwargs):
            begin=text.index(',"verdict":')-1;end=text.index(',"explanation":')
            ids=self.encode(text[:begin])+[12345]+self.encode(text[end:])
            return dict(input_ids=ids,offset_mapping=[(i,i+1) for i in range(begin)]+[(begin,end)]+[(i,i+1) for i in range(end,len(text))])
    processor=Processor();processor.tokenizer=MergedTokenizer()
    row=dict(case_id="boundary",input=fixture_payload(),target=draft_response(fixture_payload()),
             supervision=dict(claims=True,verdict=False,explanation=True))
    encoded=encode_training_row(processor,row,max_length=10000)
    assert encoded["labels"][encoded["input_ids"].index(12345)]==-100


def test_rows_without_any_grounded_target_are_not_training_examples():
    from src.qwen_training import has_supervised_target
    row=dict(target=dict(claims=None,verdict=None,explanation=None),supervision=dict(claims=True,verdict=True,explanation=True))
    assert not has_supervised_target(row)
    row["target"]["claims"]=[]
    assert has_supervised_target(row)  # a known empty list is a real target


def test_calibration_refuses_test_before_loading_data(tmp_path):
    with pytest.raises(ValueError,match="only on validation"):
        calibrate_scores(tmp_path/"missing.json",None,tmp_path/"out.json",split="test")


def test_unknown_null_target_is_masked_even_with_mistaken_flag():
    row=dict(case_id="fixture",input=fixture_payload(),target=draft_response(fixture_payload()),
             supervision=dict(claims=True,verdict=True,explanation=True))
    row["target"]["verdict"]=None
    encoded=encode_training_row(Processor(),row,max_length=10000)
    assert all(v==-100 for v,r in zip(encoded["labels"],encoded["roles"]) if r=="verdict")


def test_empty_evidence_schema_cannot_invent_unused_citations():
    from src.claims import response_schema
    payload=fixture_payload();payload["evidence"]=[]
    assert response_schema(payload)["properties"]["claims"]["maxItems"]==0


class FrozenEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(32,1024)
        self.embedding.weight.requires_grad_(False)
    def get_input_embeddings(self):return self.embedding


def test_soft_prefix_mask_positions_padding_and_gradient():
    model=FrozenEmbedding();connector=SoftConnector()
    batch=dict(input_ids=torch.tensor([[2,3,4,0],[2,3,0,0]]),attention_mask=torch.tensor([[1,1,1,0],[1,1,0,0]]))
    labels=torch.tensor([[-100,3,4,-100],[-100,3,-100,-100]])
    features=torch.randn(2,24,262)
    result,shifted=prepend_soft_tokens(model,batch,labels,connector,features)
    assert result["inputs_embeds"].shape==(2,20,1024)
    assert (shifted[:,:16]==-100).all()
    assert torch.equal(shifted[:,16:],labels)
    assert (result["attention_mask"][:,:16]==1).all()
    assert result["position_ids"][0,16].item()==16 and result["position_ids"][0,-1].item()==0
    result["inputs_embeds"].square().sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in connector.parameters())>0
    assert model.embedding.weight.grad is None
    empty,masked=prepend_soft_tokens(model,batch,labels,connector,torch.empty(0,262))
    assert (empty["attention_mask"][:,:16]==0).all()
    assert empty["position_ids"][0,16].item()==0
    assert torch.equal(swap_feature_sides(swap_feature_sides(features)),features)


def test_cache_checks_payload_checkpoint_audio_and_preprocessing(tmp_path):
    audio=tmp_path/"source.wav";audio.write_bytes(b"original-audio")
    checkpoint=tmp_path/"head.pt";checkpoint.write_bytes(b"frozen-head")
    path=tmp_path/"cache.npz";np.savez_compressed(path,features=np.zeros((4,262),np.float32))
    payload=fixture_payload()
    meta=dict(payload_sha256=digest(payload),features_sha256=file_digest(path),
        preprocessing=dict(definition=FEATURE_DEFINITION),checkpoint=dict(path=str(checkpoint),sha256=file_digest(checkpoint)),
        audio=dict(a=dict(path=str(audio),sha256=file_digest(audio))))
    path.with_suffix(".json").write_text(json.dumps(meta))
    assert load_feature_cache(path,payload)[0].shape==(4,262)
    with pytest.raises(ValueError,match="evidence fingerprint"):load_feature_cache(path,{})
    audio.write_bytes(b"changed-audio")
    with pytest.raises(ValueError,match="audio fingerprint"):load_feature_cache(path,payload)
    audio.write_bytes(b"original-audio");checkpoint.write_bytes(b"changed-head")
    with pytest.raises(ValueError,match="checkpoint fingerprint"):load_feature_cache(path,payload)


def test_metrics_do_not_credit_fallback_or_invent_semantic_accuracy():
    from src.qwen_metrics import aggregate_metrics
    result=dict(condition="B",base_case_id="base",variant="original",success=False,result=None,
        validation=dict(valid=False,automatic_errors=[dict(code="high_verdict_needs_multiple_factors")],claims=[dict(supported=True)]),
        attempts=[dict(response=dict(),validation=dict(valid=False,automatic_errors=[]),input_tokens=2,output_tokens=3,input_processing_seconds=.1,generation_seconds=.2)],
        metadata=dict(input_tokens=2,output_tokens=3,model_load_seconds=1,tokenizer_cache_seconds=.1),evaluation_truth={})
    m=aggregate_metrics([result],["B"])["B"]
    assert m["fallback_rate"]==1 and m["successful_cases"]==0
    assert m["judgment_accuracy"] is None and m["human_semantic_supported_claim_rate"] is None
    assert m["automatically_supported_claim_rate"]==1 and m["automatic_contradiction_response_rate"]==1
    assert m["fallback_required_rate"]==1 and m["fallback_used_rate"]==0


def configured_comparison(tmp_path):
    import soundfile as sf
    from src.backends import DSPBackend
    from src.compare import analyze_song,compare_songs
    path=tmp_path/"tone.wav";sf.write(path,.1*np.sin(2*np.pi*440*np.arange(24000)/24000),24000)
    backend=DSPBackend();song=analyze_song(path,backend)
    return compare_songs(song,song,backend)


def test_parent_revalidates_worker_success_and_records_real_fallback(tmp_path,monkeypatch):
    from src.qwen_integration import judge_configured
    from src.judge import rule_judge
    comparison=configured_comparison(tmp_path)
    def fake_run(*args,**kwargs):
        payload=json.loads(kwargs["input"])["payload"];bad=draft_response(payload);bad["claims"][0]["evidence_ids"]=["E999"]
        return SimpleNamespace(stdout=json.dumps(dict(results=[dict(success=True,result=bad,validation=dict(valid=True),
            attempts=[dict(raw_response=json.dumps(bad),validation=dict(valid=True))],metadata={})])))
    monkeypatch.setattr("src.qwen_integration.subprocess.run",fake_run)
    report=judge_configured(comparison,"qwen","fake-python","cpu",{})
    assert report["execution"]["status"]=="failed_rule_fallback" and report["execution"]["fallback_used"]
    assert not report["model_validation"]["valid"] and report["claim_validation"]["valid"]
    assert report["rule_baseline"]==rule_judge(comparison)
    assert "E999" in report["generation_trace"]["attempts"][0]["raw_response"]


def test_retry_trace_and_original_times_survive_worker_boundary(tmp_path,monkeypatch):
    from src.qwen_integration import judge_configured
    comparison=configured_comparison(tmp_path)
    def fake_run(*args,**kwargs):
        payload=json.loads(kwargs["input"])["payload"];response=draft_response(payload)
        return SimpleNamespace(stdout=json.dumps(dict(results=[dict(success=True,result=response,validation=dict(valid=True),
            attempts=[dict(raw_response="bad-json",validation=dict(valid=False)),dict(raw_response=json.dumps(response),validation=dict(valid=True))],metadata={})])))
    monkeypatch.setattr("src.qwen_integration.subprocess.run",fake_run)
    report=judge_configured(comparison,"qwen","fake-python","cpu",{})
    assert report["execution"]["status"]=="executed_validated" and not report["execution"]["fallback_used"]
    assert len(report["generation_trace"]["attempts"])==2
    assert report["selected"]["cited_intervals"][0]["a"]==comparison["evidence"][0]["a"]


def test_human_semantics_require_exact_output_and_real_claim_ids():
    from src.qwen_metrics import validate_semantic_review
    response=draft_response(fixture_payload());raw=json.dumps(response,ensure_ascii=False)
    result=dict(attempts=[dict(raw_response=raw,response=response)])
    review=dict(review_status="human_approved",reviewer="test-reviewer",response_sha256=digest(raw),supported_claim_ids=["C001"],unsupported_claim_ids=[],contradiction=False)
    assert validate_semantic_review(review,result)
    with pytest.raises(ValueError,match="claim IDs"):validate_semantic_review({**review,"supported_claim_ids":["C999"]},result)
    with pytest.raises(ValueError,match="hash"):validate_semantic_review({**review,"response_sha256":"stale"},result)
    assert not validate_semantic_review(dict(review_status="unreviewed"),result)


def test_audio_counterfactual_preserves_source_and_unedited_regions():
    from src.qwen_audio_eval import zero_audio_region
    x=np.linspace(-.2,.2,24000*3,dtype=np.float32);before=x.copy()
    y=zero_audio_region(x,1.,2.)
    assert np.array_equal(x,before)
    assert np.array_equal(y[:24000],x[:24000]) and np.array_equal(y[48000:],x[48000:])
    assert not y[24000:48000].any()
    with pytest.raises(ValueError):zero_audio_region(x,2.,4.)

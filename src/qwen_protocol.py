"""Shared train/inference serialization, without importing Transformers."""
import json
import copy

CLAIM_PROMPT = """두 곡의 음악적 유사성을 검토하세요. 입력은 분석기의 예측 또는 프록시이며 정답이 아닙니다.
JSON 하나로 claims, verdict, explanation을 출력하세요. verdict는 high_similarity, review,
low_similarity, insufficient_evidence 중 하나입니다. 원시 코사인은 확률이나 분석 정확도가 아닙니다.
available_claims는 근거 ID와 요소로 묶은 확인 가능한 주장 유형과 문장입니다. 사용하는 주장만 선택하고
claim_id를 C001부터 붙이세요. evidence_ids, factor, role, predicate가 실제 근거와 맞아야 합니다.
수치를 쓰면 assertion에 quantity, value, operator를 명시하세요. 근거가 불확실하면 limits 또는
insufficient 주장을 포함하세요. 구조 기능 라벨은 맥락이며 음악 내용의 일치를 입증하지 않습니다.
높은 유사성 판단에는 서로 다른 요소의 지지가 필요합니다. 보정·신뢰도 한계를 함께 해석하세요.
explanation은 선택한 주장과 일치하는 짧은 한국어 두 문장으로 작성하세요. 숫자와 시간을 다시
쓰지 말고, 입력에 없는 음악적 사실이나 표절 여부를 단정하지 마세요. 판단과 모순되는 결론을 쓰지 마세요.
"""


def prompt_content(payload):
    from .claims import claim_catalog
    from .evidence import digest
    # The catalog is a deterministic evidence interface, not a gold answer.
    catalog={}
    for claim in claim_catalog(payload):
        if claim["predicate"]=="observed_value":continue
        for eid in claim["evidence_ids"]:
            catalog.setdefault(eid,{}).setdefault(claim["factor"],[]).append(
                {k:claim[k] for k in ("role","predicate","statement")})
    compact=copy.deepcopy(payload)
    backend=compact["backend"];training=backend.get("training",{})
    compact["backend"]={k:backend[k] for k in ("backend","checkpoint","backbone","heads","status") if k in backend}
    compact["backend"].update(identity_sha256=digest(backend),
        backbone_model=training.get("model_versions",{}).get("mert95"),
        training_scope={k:training[k] for k in ("smoke","prediction_weight","epochs_completed") if k in training})
    # Missing factors have no score, persistence or calibrated threshold to
    # interpret. Their missing state and head identity remain explicit.
    for evidence in compact["evidence"]:
        for name,value in list(evidence["factors"].items()):
            if value["status"]=="unavailable":
                evidence["factors"][name]={k:value[k] for k in ("status","name_ko","raw_cosine","head","provenance")}
    return json.dumps(dict(evidence=compact, available_claims=catalog),ensure_ascii=False,separators=(",",":"))


def messages_for(payload):
    return [{"role":"system","content":[{"type":"text","text":CLAIM_PROMPT}]},
            {"role":"user","content":[{"type":"text","text":prompt_content(payload)}]}]

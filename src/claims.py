"""Machine-checkable claim predicates, with separate human semantic review."""
import copy
import math
import re

from .evidence import FACTOR_NAMES

VERDICTS = ("high_similarity", "review", "low_similarity", "insufficient_evidence")
ROLES = ("supports", "limits", "insufficient")
PREDICATES = ("score_high", "score_low", "uncalibrated_score", "analyzer_unvalidated",
              "information_missing", "persistent_match", "conflicting_windows",
              "same_section_context", "different_section_context", "observed_value",
              "temporal_pattern_match", "temporal_pattern_difference")


def claim_catalog(payload):
    """Only these tuples are supported; statement paraphrases still need review."""
    catalog = []
    for e in payload["evidence"]:
        eid = e["evidence_id"]
        for factor, value in e["factors"].items():
            name = FACTOR_NAMES[factor]
            def add(predicate, role, statement, assertion=None):
                c = dict(evidence_ids=[eid], factor=factor, role=role, predicate=predicate, statement=statement)
                if assertion is not None:
                    c["assertion"] = assertion
                catalog.append(c)
            if value["status"] == "unavailable":
                add("information_missing", "insufficient", f"{name}의 유사성을 판단할 정보가 없습니다.")
                continue
            score, threshold = value["raw_cosine"], value["interpretation_threshold"]
            if score >= threshold:
                add("score_high", "supports", f"{name}의 원시 유사도가 설정된 해석 기준 이상입니다.")
            else:
                add("score_low", "limits", f"{name}의 원시 유사도가 설정된 해석 기준보다 낮습니다.")
            add("observed_value", "supports" if score >= threshold else "limits",
                f"{name}의 원시 코사인은 {score:.4f}입니다.", dict(quantity="raw_cosine", value=round(score, 4), operator="eq"))
            if value["calibration"].get("status") == "uncalibrated":
                add("uncalibrated_score", "limits", f"{name} 점수는 보정되지 않아 판단의 확실성으로 해석할 수 없습니다.")
            if value["analyzer_reliability"]["status"] != "validated":
                add("analyzer_unvalidated", "limits", f"{name} 분석기의 정확도는 이 자료에서 검증되지 않았습니다.")
            persistence = value["persistence"]
            if persistence["n_windows"] >= 2 and persistence["above_threshold"] == persistence["n_windows"]:
                add("persistent_match", "supports", f"대응하는 여러 창에서 {name}의 높은 점수가 이어집니다.")
            if persistence["mismatches"]:
                add("conflicting_windows", "limits", f"대응 창 중 {name}의 점수가 낮은 부분도 있습니다.")
            temporal = e.get("temporal_comparison",{}).get(factor,{})
            if temporal.get("status")=="available":
                match = temporal["derived_agreement"] >= temporal["interpretation_threshold"]
                add("temporal_pattern_match" if match else "temporal_pattern_difference", "supports" if match else "limits",
                    f"시간순 요약의 {name} 패턴이 근사 비교에서 대응합니다. 예측 또는 프록시 요약이라는 한계가 있습니다." if match else
                    f"시간순 요약의 {name} 패턴에는 차이가 관찰됩니다. 원시 임베딩 점수와 함께 검토해야 합니다.")
        a = {s["label"] for s in e["structure_context"].get("a", [])}
        b = {s["label"] for s in e["structure_context"].get("b", [])}
        if a and b:
            same = bool(a & b)
            catalog.append(dict(evidence_ids=[eid], factor="phrase_structure", role="limits",
                predicate="same_section_context" if same else "different_section_context",
                statement="구조 기능 라벨의 일치는 맥락이며 음악 내용의 일치를 입증하지 않습니다." if same else
                          "두 구간의 구조 기능 라벨이 다르므로 곡 안에서의 역할을 구분해 검토해야 합니다."))
    return catalog


def response_schema(payload, max_claims=4):
    ids = [e["evidence_id"] for e in payload["evidence"]]
    return dict(type="object", properties=dict(
        claims=dict(type="array", maxItems=max_claims if ids else 0, items=dict(type="object", properties=dict(
            claim_id=dict(type="string", enum=[f"C{i:03}" for i in range(1, max_claims+1)]),
            evidence_ids=dict(type="array", items=dict(type="string", enum=ids or ["unused"]), minItems=1, maxItems=max(1,len(ids))),
            factor=dict(type="string", enum=list(FACTOR_NAMES)), role=dict(type="string", enum=list(ROLES)),
            predicate=dict(type="string", enum=list(PREDICATES)), statement=dict(type="string", minLength=1, maxLength=140),
            assertion=dict(type="object", properties=dict(quantity=dict(type="string", enum=["raw_cosine", "time_start", "time_end"]),
                value=dict(type="number"), operator=dict(type="string", enum=["eq", "ge", "le"]),
                side=dict(type="string", enum=["a", "b"])), required=["quantity", "value", "operator"], additionalProperties=False)),
            required=["claim_id", "evidence_ids", "factor", "role", "predicate", "statement"], additionalProperties=False)),
        verdict=dict(type="string", enum=list(VERDICTS)), explanation=dict(type="string", minLength=1, maxLength=300)),
        required=["claims", "verdict", "explanation"], additionalProperties=False)


def validate_claim_response(value, payload):
    errors, checked, human = [], [], []
    def error(code, claim_id=None, detail=None):
        errors.append(dict(code=code, claim_id=claim_id, detail=detail))
    if not isinstance(value, dict) or set(value) != {"claims", "verdict", "explanation"}:
        return dict(valid=False, automatic_errors=[dict(code="schema_fields")], claims=[], human_review_required=[])
    if not isinstance(value["verdict"], str) or value["verdict"] not in VERDICTS:
        error("invalid_verdict")
    if not isinstance(value["claims"], list):
        return dict(valid=False, automatic_errors=[dict(code="claims_not_array")], claims=[], human_review_required=[])
    catalog = claim_catalog(payload)
    allowed = {(tuple(c["evidence_ids"]), c["factor"], c["role"], c["predicate"]) for c in catalog}
    byid = {e["evidence_id"]:e for e in payload["evidence"]}
    seen = set()
    for claim in value["claims"]:
        before = len(errors)
        required={"claim_id","evidence_ids","factor","role","predicate","statement"}
        if not isinstance(claim, dict) or not required <= set(claim) or set(claim)-required-{"assertion"}:
            error("claim_schema"); continue
        cid, ids, factor = claim["claim_id"], claim["evidence_ids"], claim["factor"]
        if not isinstance(cid, str) or not re.fullmatch(r"C\d{3}", cid) or cid in seen:
            error("duplicate_or_invalid_claim_id", str(cid))
        seen.add(str(cid))
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
            error("invalid_evidence_ids", cid); continue
        if len(set(ids)) != len(ids):
            error("duplicate_evidence_ids", cid)
        if not isinstance(factor,str) or factor not in FACTOR_NAMES:
            error("unknown_factor", cid); continue
        if not isinstance(claim["role"],str) or not isinstance(claim["predicate"],str):
            error("invalid_predicate_or_role",cid);continue
        if any(eid not in byid for eid in ids):
            error("unknown_evidence_id", cid)
        elif any(((eid,), factor, claim["role"], claim["predicate"]) not in allowed for eid in ids):
            error("unsupported_predicate_or_role", cid)
        text = claim["statement"]
        if not isinstance(text, str) or not re.search("[가-힣]", text):
            error("statement_not_korean", cid)
            text = ""
        assertions = claim.get("assertion")
        if claim["predicate"] == "observed_value" and not assertions:
            error("numerical_assertion_required", cid)
        if assertions:
            try:
                quantity, expected, operator = assertions["quantity"], float(assertions["value"]), assertions["operator"]
                if not math.isfinite(expected) or operator not in ("eq", "ge", "le"):
                    raise ValueError()
                for eid in ids:
                    if quantity == "raw_cosine":
                        actual = byid[eid]["factors"][factor]["raw_cosine"]
                    elif quantity in ("time_start", "time_end"):
                        actual = byid[eid][assertions["side"]][quantity.removeprefix("time_")]
                    else:
                        raise ValueError()
                    ok = actual is not None and (abs(actual-expected) <= 1e-4 if operator == "eq" else actual >= expected if operator == "ge" else actual <= expected)
                    if not ok:
                        error("numeric_direction_or_time_mismatch", cid)
            except (KeyError, TypeError, ValueError):
                error("invalid_numeric_assertion", cid)
            if isinstance(assertions,dict):
                if assertions.get("operator")=="ge" and re.search(r"이하|미만|작습니다",text):error("numeric_operator_statement_conflict",cid)
                if assertions.get("operator")=="le" and re.search(r"이상|초과|큽니다",text):error("numeric_operator_statement_conflict",cid)
        # Any numeric prose requires the independently checked structured value.
        prose = re.sub(r"[EC]\d{3}", "", text)
        numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", prose)
        if numbers:
            try:
                if not assertions or any(abs(float(n)-float(assertions["value"])) > 1e-4 for n in numbers):error("unbound_number_in_statement",cid)
            except (TypeError,ValueError,KeyError):error("unbound_number_in_statement",cid)
        if factor == "phrase_structure" and re.search(r"(?:후렴|벌스|구조).*같.*(?:따라서|때문).*유사|내용.*일치(?:합니다|한다)", text):
            error("structure_used_as_content", cid)
        if claim["predicate"] == "score_high" and re.search(r"유사도.*낮|점수.*낮|기준.*미만", text):
            error("statement_predicate_direction_conflict", cid)
        if claim["predicate"] == "score_low" and re.search(r"유사도.*높|점수.*높|기준.*이상", text):
            error("statement_predicate_direction_conflict", cid)
        if re.search(r"(?:확률|정확도).*\d|검증된\s*(?:주선율|베이스|코드)", text):
            error("unsupported_reliability_claim", cid)
        checked.append(dict(claim_id=cid, supported=len(errors)==before, predicate=claim["predicate"]))
        human.append(dict(claim_id=cid, check="statement_semantics_and_provenance_interpretation", status="unreviewed"))
    valid_claims=[c for c in value["claims"] if isinstance(c,dict)]
    supports = {str(c.get("factor")) for c in valid_claims if c.get("role")=="supports" and c.get("predicate") not in ("same_section_context", "different_section_context")}
    if value["verdict"] == "high_similarity" and len(supports) < 2:
        error("high_verdict_needs_multiple_factors")
    opposing=any(c.get("role")=="limits" and c.get("predicate") in ("score_low","temporal_pattern_difference","conflicting_windows") for c in valid_claims)
    if value["verdict"] == "low_similarity" and supports and not opposing:
        error("low_verdict_conflicts_with_support_claims")
    if value["verdict"] == "insufficient_evidence" and supports:
        error("insufficient_verdict_conflicts_with_support_claims")
    if value["verdict"] in ("high_similarity", "review") and not value["claims"]:
        error("verdict_without_claims")
    if supports and not any(c.get("role")=="limits" and c.get("predicate") in ("uncalibrated_score","analyzer_unvalidated") for c in valid_claims):
        error("missing_uncertainty_claim")
    explanation = value["explanation"]
    if not isinstance(explanation, str) or not re.search("[가-힣]", explanation):
        error("explanation_not_korean"); explanation=""
    if re.search(r"\d", re.sub(r"[EC]\d{3}", "", explanation)):
        error("numeric_explanation_requires_original_evidence_display")
    if value["verdict"] == "high_similarity" and re.search(r"(?:high_similarity|높은\s*유사성?).{0,25}(?:충족되지|아니|없다고|어렵)", explanation):
        error("verdict_explanation_polarity_conflict")
    cited = {c["claim_id"] for c in value["claims"] if isinstance(c, dict) and "claim_id" in c}
    if not set(re.findall(r"C\d{3}", explanation)) <= cited:
        error("unknown_claim_in_explanation")
    human.append(dict(check="free_text_contradictions_and_unstated_musical_facts", status="unreviewed"))
    return dict(valid=not errors, automatic_errors=errors, claims=checked,
                human_review_required=human, semantic_accuracy="not_measured_by_automatic_validation")


def draft_response(payload, max_claims=4):
    """Auditable rule labels for pilot data ONLY, never human ground truth."""
    catalog = claim_catalog(payload)
    supports = [c for c in catalog if c["predicate"] == "score_high"]
    low = [c for c in catalog if c["predicate"] == "score_low"]
    limits = [c for c in catalog if c["predicate"] in ("uncalibrated_score", "analyzer_unvalidated")]
    missing = [c for c in catalog if c["predicate"] == "information_missing"]
    chosen = (supports[:max(1,max_claims-1)] + limits[:1]) if supports else (low[:max(1,max_claims-1)] + limits[:1]) if low else missing[:max_claims]
    claims = [dict(claim_id=f"C{i+1:03}", **copy.deepcopy(c)) for i,c in enumerate(chosen)]
    verdict = "review" if supports or low else "insufficient_evidence"
    text = "대응 구간의 일부 요소에서 높은 원시 유사도가 관찰됩니다. 점수 보정과 분석 정확도에 한계가 있어 추가 검토가 필요합니다." if supports else (
        "관측된 점수에서 뚜렷한 대응을 확인하기 어렵습니다. 분석의 한계와 누락된 정보를 함께 검토해야 합니다." if low else "현재 음악 근거가 부족해 종합 판단을 보류합니다.")
    return dict(claims=claims, verdict=verdict, explanation=text)


def legacy_projection(response):
    return dict(judgment=response["verdict"], evidence_ids=list(dict.fromkeys(eid for c in response["claims"] for eid in c["evidence_ids"])),
                explanation=response["explanation"])

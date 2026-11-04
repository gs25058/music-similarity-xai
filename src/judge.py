import copy
import json
import re
import subprocess

JUDGMENTS = {"high_similarity", "review", "low_similarity", "insufficient_evidence"}


def validate_judgment(value, evidence):
    if not isinstance(value, dict) or set(value) != {"judgment", "evidence_ids", "explanation"}:
        raise ValueError("Expected exactly judgment, evidence_ids, explanation")
    if value["judgment"] not in JUDGMENTS:
        raise ValueError("Unknown judgment")
    ids = value["evidence_ids"]
    known = {e["evidence_id"] for e in evidence}
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Invalid evidence_ids")
    if not set(ids) <= known:
        raise ValueError("Unknown evidence ID")
    text = value["explanation"]
    if not isinstance(text, str) or not re.search("[가-힣]", text) or not 1 <= len(text) <= 8000:
        raise ValueError("Expected Korean explanation")
    prose = re.sub(r"(?<![A-Za-z0-9])E\d{3,}(?![0-9])", "", text)
    if re.search(r"\d", prose):
        raise ValueError("explanation은 근거 ID 외 숫자 없이 작성하세요. 점수와 시간은 원본 근거에서 표시합니다.")
    if value["judgment"] in ("high_similarity", "review") and not ids:
        raise ValueError("Positive/review judgment requires cited evidence")
    mentioned = set(re.findall(r"(?<![A-Za-z0-9])E\d{3,}(?![0-9])", text))
    if not mentioned <= set(ids):
        raise ValueError("Explanation contains an uncited evidence ID")
    sections_observed = any(e.get("structure_context", {}).get(side) for e in evidence for side in ("a", "b"))
    if sections_observed and re.search(r"구조\s*정보를?\s*관측하지\s*못|structure_context\s+(?:is\s+)?unavailable", text):
        raise ValueError("Section labels were observed; explanation incorrectly claims missing structure context")
    if not sections_observed and re.search(r"모두\s*['\"]?(?:후렴|벌스|chorus|verse)", text):
        raise ValueError("Explanation asserts section labels absent from evidence")
    return value


def rule_judge(comparison):
    evidence = comparison["evidence"]
    if not evidence:
        # Distinguish an analyzed low match from silence/no supported features.
        has_scores = any(v is not None for v in comparison.get("matrix_maxima", {}).values())
        judgment = "low_similarity" if has_scores else "insufficient_evidence"
        return dict(judgment=judgment, evidence_ids=[], explanation=("설정된 임계값을 넘는 대응 구간이 없습니다."
                    if has_scores else "분석 가능한 음악 근거가 부족합니다."))
    best = evidence[0]
    factors = best["factors"]
    available = [v["cosine"] for k, v in factors.items() if v["cosine"] is not None and k != "timbre"]
    melody = factors["pitch_intervals"]["cosine"]
    if melody is None:
        melody = factors["melody"]["cosine"]
    high = len(available) >= 2 and melody is not None and melody >= .92 and best["combined_cosine"] >= .92
    label = "high_similarity" if high else "review"
    explanation = f"{best['evidence_id']}의 대응 구간에서 음악 요소의 유사성이 관찰되어 추가 검토가 필요합니다. "
    explanation += "여러 요소에서 높은 점수가 확인되었습니다." if high else "요소별 차이와 누락된 근거를 함께 확인해야 합니다."
    explanation += " 구조 라벨은 맥락으로만 사용했습니다."
    return dict(judgment=label, evidence_ids=[best["evidence_id"]], explanation=explanation)


def evidence_payload(evidence, limit=3):
    result = []
    for item in evidence[:limit]:
        e = copy.deepcopy(item)
        e.pop("window_pairs", None)
        e.pop("candidate_kind", None)  # implementation detail, not musical structure
        e["combined_cosine"] = round(e["combined_cosine"], 4)
        for factor in e["factors"].values():
            if factor["cosine"] is not None:
                factor["cosine"] = round(factor["cosine"], 4)
        # Preserve missing states and predictions; bound repeated per-window information.
        e["music_information"] = {side: e["music_information"][side][:1] for side in ("a", "b")}
        if not e["structure_context"]["a"] and not e["structure_context"]["b"]:
            e["structure_context"] = dict(status="unavailable", a=[], b=[],
                                          missing="No observed section labels or structural similarity")
        result.append(e)
    return result


def judge_pair(comparison, backend="rule", qwen_python=None, device="cpu"):
    baseline = validate_judgment(rule_judge(comparison), comparison["evidence"])
    selected, status = baseline, dict(backend="rule", status="executed")
    if backend == "qwen":
        request = dict(evidence=evidence_payload(comparison["evidence"]),
                       matrix_maxima=comparison.get("matrix_maxima", {}), device=device)
        if not qwen_python:
            raise ValueError("--qwen-python must point to the isolated Qwen Python environment")
        try:
            process = subprocess.run([qwen_python, "scripts/qwen_worker.py"], input=json.dumps(request),
                                     text=True, capture_output=True, check=True, timeout=900)
            response = json.loads(process.stdout)
            selected = validate_judgment(response["result"], request["evidence"])
            status = dict(backend="qwen", status="executed_validated", metadata=response.get("metadata", {}))
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
            # Explicit fallback, with validation failure and raw model output retained.
            detail = getattr(exc, "stderr", None) or str(exc)
            status = dict(backend="qwen", status="failed_rule_fallback", error=detail[-5000:])
            if 'process' in locals():
                status["raw_output"] = process.stdout[-12000:]
    by_id = {e["evidence_id"]: e for e in comparison["evidence"]}
    selected = {**selected, "cited_intervals": [dict(evidence_id=eid, a=by_id[eid]["a"], b=by_id[eid]["b"])
                                               for eid in selected["evidence_ids"]]}
    return dict(selected=selected, rule_baseline=baseline, execution=status)


def swap_comparison(comparison):
    swapped = copy.deepcopy(comparison)
    for e in swapped["evidence"]:
        e["a"], e["b"] = e["b"], e["a"]
        for key in ("music_information", "structure_context"):
            e[key]["a"], e[key]["b"] = e[key]["b"], e[key]["a"]
    return swapped

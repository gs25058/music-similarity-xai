import copy
import json
import re
import subprocess

JUDGMENTS = {"high_similarity", "review", "low_similarity", "insufficient_evidence"}


def validate_judgment(value, evidence):
    if not isinstance(value, dict) or set(value) != {"judgment", "evidence_ids", "explanation"}:
        raise ValueError("Expected exactly judgment, evidence_ids, explanation")
    if not isinstance(value["judgment"], str) or value["judgment"] not in JUDGMENTS:
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
    letters = sum(c.isalpha() for c in text)
    if len(re.findall("[가-힣]", text)) < .25 * letters:
        raise ValueError("설명은 한국어 문장으로 작성하세요.")
    if re.search(r"(\S+)(?:\s+\1){5,}|([^\s])\2{9,}|(.{2,50}?)(?:\3){4,}", text):
        raise ValueError("같은 글자나 문구를 반복하지 말고 음악적 근거를 짧게 설명하세요.")
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


def music_information_summary(info):
    """Serialize readable observations, preserving uncertainty and provenance."""
    result = {key: info[key] for key in ("status", "source", "pitch_source", "accompaniment_source",
              "harmony_source", "training_accompaniment_sources", "missing", "reason",
              "dominant_pitch_midi_median", "low_register_midi_median") if key in info}
    if "onset_times_seconds" in info:
        result["observed_onset_count"] = len(info["onset_times_seconds"])
    if "chroma_mean" in info:
        result["chroma_peak_pitch_classes"] = sorted(range(12), key=lambda i: info["chroma_mean"][i], reverse=True)[:3]
    if "predictions" in info:
        predictions = {}
        for factor, tasks in info["predictions"].items():
            summary = {}
            for task, value in tasks.items():
                if isinstance(value, dict) and "class_histogram" in value:
                    counts = value["class_histogram"]
                    summary[task] = dict(most_frequent_predicted_class=max(range(len(counts)), key=counts.__getitem__))
                elif isinstance(value, list):
                    summary[task] = dict(peak_pitch_classes=sorted(range(len(value)), key=value.__getitem__, reverse=True)[:3])
                else:
                    summary[task] = value
            predictions[factor] = summary
        result["predictions"] = predictions
    return result


def evidence_payload(evidence, limit=3):
    result = []
    for item in evidence[:limit]:
        e = copy.deepcopy(item)
        e.pop("window_pairs", None)
        e.pop("candidate_kind", None)  # implementation detail, not musical structure
        e["combined_cosine"] = round(e["combined_cosine"], 4)
        names = {"pitch_intervals": "음정 진행", "melody": "선율", "rhythm": "리듬", "harmony": "화성",
                 "accompaniment_bass": "반주·베이스", "phrase_structure": "프레이즈 구조", "timbre": "음색"}
        for name, factor in e["factors"].items():
            factor["name_ko"] = names[name]
            if factor["cosine"] is not None:
                factor["cosine"] = round(factor["cosine"], 4)
                factor["score_band_ko"] = ("매우 높음" if factor["cosine"] >= .92 else
                                           "높음" if factor["cosine"] >= .8 else "부분적" if factor["cosine"] >= .5 else "낮음")
            else:
                factor["score_band_ko"] = "판단 정보 없음"
        # Preserve missing states and predictions; bound repeated per-window information.
        e["music_information"] = {side: [music_information_summary(info) for info in e["music_information"][side][:1]] for side in ("a", "b")}
        if not e["structure_context"]["a"] and not e["structure_context"]["b"]:
            e["structure_context"] = dict(status="unavailable", a=[], b=[],
                                          missing="No observed section labels or structural similarity")
        result.append(e)
    return result


def judge_pair(comparison, backend="rule", qwen_python=None, device="cpu", config=None, context=None):
    if config is not None:
        from .qwen_integration import judge_configured
        return judge_configured(comparison,backend,qwen_python,device,config,context)
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
    status["fallback_used"]=backend=="qwen" and status["status"]=="failed_rule_fallback"
    selected = {**selected, "cited_intervals": [dict(evidence_id=eid, a=by_id[eid]["a"], b=by_id[eid]["b"])
                                               for eid in selected["evidence_ids"]]}
    return dict(selected=selected, rule_baseline=baseline, execution=status)


def swap_comparison(comparison):
    swapped = copy.deepcopy(comparison)
    for e in swapped["evidence"]:
        e["a"], e["b"] = e["b"], e["a"]
        for key in ("music_information", "structure_context"):
            e[key]["a"], e[key]["b"] = e[key]["b"], e[key]["a"]
        context = e["structure_context"]
        if "a_status" in context and "b_status" in context:
            context["a_status"], context["b_status"] = context["b_status"], context["a_status"]
    return swapped

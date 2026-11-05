"""Score stored outputs without regenerating them; semantic labels are external."""
from .claims import legacy_projection,VERDICTS
from .evidence import digest


def validate_semantic_review(review,result):
    status=review.get("review_status","human_approved" if review.get("reviewer") else "unreviewed")
    if status=="unreviewed":return False
    if status!="human_approved" or not review.get("reviewer"):raise ValueError("Semantic review needs approval and a reviewer")
    if review["response_sha256"]!=digest(result["attempts"][-1]["raw_response"]):raise ValueError("Semantic review output hash mismatch")
    value=result["attempts"][-1].get("response")
    if not isinstance(value,dict):value={}
    known={c["claim_id"] for c in (value.get("claims") or []) if isinstance(c,dict) and isinstance(c.get("claim_id"),str)}
    good,bad=review.get("supported_claim_ids",[]),review.get("unsupported_claim_ids",[])
    if len(set(good))!=len(good) or len(set(bad))!=len(bad) or set(good)&set(bad) or not (set(good)|set(bad))<=known:
        raise ValueError("Unknown/duplicate/conflicting semantic claim IDs")
    contradiction=review.get("contradiction")
    if contradiction is not None and not isinstance(contradiction,bool):raise ValueError("Contradiction annotation must be boolean or null")
    text=value.get("explanation","")
    for span in review.get("statement_spans",[]):
        if not (0<=span["start"]<span["end"]<=len(text)) or not isinstance(span["supported"],bool):raise ValueError("Invalid semantic text span")
    if not good and not bad and contradiction is None and not review.get("statement_spans"):raise ValueError("Approved semantic review has no annotations")
    return True


def mean(values):
    values=[v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def judgment_metrics(pairs):
    if not pairs:return dict(n=0,accuracy=None,macro_f1=None)
    f1=[]
    for label in VERDICTS:
        tp=sum(a==label and b==label for a,b in pairs)
        fp=sum(a==label and b!=label for a,b in pairs)
        fn=sum(a!=label and b==label for a,b in pairs)
        if 2*tp+fp+fn:f1.append(2*tp/(2*tp+fp+fn))
    return dict(n=len(pairs),accuracy=mean([a==b for a,b in pairs]),macro_f1=mean(f1))


def aggregate_metrics(all_results,conditions):
    metrics={}
    for condition in conditions:
        results=[r for r in all_results if r["condition"]==condition]
        rows=[r for r in results if not r["variant"].startswith("soft_")]
        valid=[r for r in rows if r["success"]]
        parsed=[r for r in rows if isinstance(r["attempts"][-1].get("response"),dict)]
        def projection(r):return r["result"] if condition=="A" else legacy_projection(r["result"])
        originals={r["base_case_id"]:r for r in rows if r["variant"]=="original"}
        cores={r["base_case_id"]:r for r in rows if r["variant"]=="unique_core"}
        changes={}
        for variant in ("swap_ab","reverse_evidence","add_irrelevant","remove_unique_core"):
            checks=[];ids=[];reaction=[];trivial=0
            for r in rows:
                if r["variant"]!=variant:continue
                if variant=="remove_unique_core":
                    checks.append(projection(r)["judgment"]=="insufficient_evidence" if r["success"] else False)
                    base=cores.get(r["base_case_id"])
                    if base is None and r.get("original_evidence_count")==1:base=originals.get(r["base_case_id"])
                    if base and base["success"] and projection(base)["judgment"]!="insufficient_evidence":
                        reaction.append(r["success"] and projection(r)["judgment"]=="insufficient_evidence")
                else:
                    base=originals.get(r["base_case_id"])
                    if base:
                        if variant=="reverse_evidence" and r.get("input_sha256") is not None and r["input_sha256"]==base.get("input_sha256"):
                            trivial+=1;continue
                        both=r["success"] and base["success"]
                        checks.append(projection(r)["judgment"]==projection(base)["judgment"] if both else False)
                        ids.append(set(projection(r)["evidence_ids"])==set(projection(base)["evidence_ids"]) if both else False)
            changes[variant]=dict(n=len(checks),valid_and_appropriate_rate=mean(checks),citation_set_consistency=mean(ids),
                trivial_identical_input_n=trivial,
                eligible_nonabstaining_reference_n=len(reaction),changed_to_abstention_rate=mean(reaction))
        claims=[c for r in rows for c in r["validation"].get("claims",[])]
        citation_errors={"unknown_evidence_id","unsupported_predicate_or_role","numeric_direction_or_time_mismatch","numeric_operator_statement_conflict","unbound_number_in_statement","unsupported_reliability_claim","structure_used_as_content"}
        contradictions={"verdict_explanation_polarity_conflict","low_verdict_conflicts_with_support_claims","high_verdict_needs_multiple_factors","statement_predicate_direction_conflict","insufficient_verdict_conflicts_with_support_claims"}
        human=[r for r in rows if r.get("human_semantic_review")]
        semantic_fractions=[]
        for r in human:
            review=r["human_semantic_review"]
            supported=review.get("supported_claim_ids",[]);unsupported=review.get("unsupported_claim_ids",[])
            if supported or unsupported:semantic_fractions.append(len(supported)/(len(supported)+len(unsupported)))
        bysource={}
        for source in ("human_reviewed","controlled_pattern_reuse","rule_provisional"):
            gold=[r for r in valid if r["evaluation_truth"].get("verdict") in VERDICTS and r["evaluation_truth"].get("verdict_source")==source]
            bysource[source]=judgment_metrics([(projection(r)["judgment"],r["evaluation_truth"]["verdict"]) for r in gold])
        human_gold=bysource["human_reviewed"]
        metrics[condition]=dict(requested_cases=len(rows),successful_cases=len(valid),parsed_response_n=len(parsed),
            judgment_accuracy=human_gold["accuracy"],judgment_macro_f1=human_gold["macro_f1"],judgment_gold_n=human_gold["n"],judgment_by_label_source=bysource,
            abstention_rate=mean([projection(r)["judgment"]=="insufficient_evidence" for r in valid]),review_rate=mean([projection(r)["judgment"]=="review" for r in valid]),
            automatically_supported_claim_rate=mean([c["supported"] for c in claims]),automatic_claim_n=len(claims),
            unsupported_or_wrong_citation_response_rate=mean([any(e["code"] in citation_errors for e in r["validation"].get("automatic_errors",[])) for r in parsed]) if condition!="A" else None,
            automatic_contradiction_response_rate=mean([any(e["code"] in contradictions for e in r["validation"].get("automatic_errors",[])) for r in parsed]) if condition!="A" else None,
            human_semantic_response_n=len(human),human_semantic_supported_claim_rate=mean(semantic_fractions),
            human_semantic_supported_span_rate=mean([s["supported"] for r in human for s in r["human_semantic_review"].get("statement_spans",[])]),
            human_contradiction_rate=mean([r["human_semantic_review"].get("contradiction") for r in human]),
            json_failure_rate=mean([any(e["code"] in ("json_parse_failure","json_or_schema_failure") for a in r["attempts"] for e in a["validation"].get("automatic_errors",[])) for r in rows]),
            first_pass_validation_rate=mean([r["attempts"][0]["validation"]["valid"] for r in rows]),
            retry_rate=mean([len(r["attempts"])>1 for r in rows]),fallback_rate=mean([not r["success"] for r in rows]),
            fallback_required_rate=mean([not r["success"] for r in rows]),
            fallback_used_rate=mean([r.get("fallback_used",False) for r in rows]),
            fallback_rate_definition="legacy alias of fallback_required_rate; quality comparison never executes the rule fallback",
            mean_input_tokens=mean([r["metadata"]["input_tokens"] for r in rows]),mean_output_tokens=mean([r["metadata"]["output_tokens"] for r in rows]),
            mean_total_input_tokens_with_retries=mean([sum(a["input_tokens"] for a in r["attempts"]) for r in rows]),
            mean_soft_prefix_slots=mean([r["attempts"][0].get("soft_prefix_tokens",0) for r in rows]),
            mean_input_embedding_positions=mean([r["attempts"][0].get("input_embedding_positions",r["metadata"]["input_tokens"]) for r in rows]),
            model_load_seconds=rows[0]["metadata"]["model_load_seconds"] if rows else None,
            tokenizer_cache_seconds=rows[0]["metadata"]["tokenizer_cache_seconds"] if rows else None,
            mean_input_processing_seconds=mean([sum(a["input_processing_seconds"] for a in r["attempts"]) for r in rows]),
            mean_generation_seconds=mean([sum(a["generation_seconds"] for a in r["attempts"]) for r in rows]),interventions=changes)
        if condition=="D":
            metrics[condition]["soft_interventions"]={}
            for change in ("remove","swap"):
                pairs=[(r,originals.get(r["base_case_id"])) for r in results if r["variant"]=="soft_"+change]
                usable=[(r,b) for r,b in pairs if b and r["success"] and b["success"]]
                parsed_pairs=[(r,b) for r,b in pairs if b and isinstance(r["attempts"][-1].get("response"),dict) and isinstance(b["attempts"][-1].get("response"),dict)]
                metrics[condition]["soft_interventions"][change]=dict(n=len(pairs),both_valid_n=len(usable),
                    verdict_changed_rate=mean([projection(r)["judgment"]!=projection(b)["judgment"] for r,b in usable]),
                    explanation_changed_rate=mean([projection(r)["explanation"]!=projection(b)["explanation"] for r,b in usable]),
                    both_parsed_n=len(parsed_pairs),
                    raw_verdict_changed_rate=mean([r["attempts"][-1]["response"].get("verdict")!=b["attempts"][-1]["response"].get("verdict") for r,b in parsed_pairs]),
                    raw_explanation_changed_rate=mean([r["attempts"][-1]["response"].get("explanation")!=b["attempts"][-1]["response"].get("explanation") for r,b in parsed_pairs]),
                    interpretation="Dependence under intervention; not proof of musical correctness. Removing prefix also changes effective context length.")
    return metrics

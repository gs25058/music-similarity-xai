"""Pipeline-prediction JSONL, conservative source groups, editable review patches."""
import copy
import json
from collections import Counter
from pathlib import Path

from .claims import draft_response, validate_claim_response
from .evidence import compact_evidence, digest, file_digest


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")


def audit_source_groups(rows):
    groups, originals={},{}
    for row in rows:
        for key, destination in ((row["source_group_id"],groups), *[(v,originals) for v in row["original_ids"]]):
            if key in destination and destination[key]!=row["split"]:
                raise ValueError(f"Source group/original leakage: {key}")
            destination[key]=row["split"]
    return dict(source_groups=len(groups), independent_originals=len(originals),
                groups_by_split={s:sum(v==s for v in groups.values()) for s in ("train","validation","test")})


def dataset_coverage(rows):
    from .claims import claim_catalog
    original=[r for r in rows if r["variant"]=="original"]
    audio_pairs=set()
    for row in original:
        path=Path(row["original_evidence"]["report_path"])
        if path.exists():
            report=json.loads(path.read_text());audio_pairs.add(tuple(report["songs"][s]["path"] for s in ("a","b")))
    return dict(**audit_source_groups(rows),rows=len(rows),base_case_views=len(original),distinct_audio_path_pairs=len(audio_pairs),
        review_counts=dict(Counter(r["review"]["status"] for r in rows)),
        original_kind_views=dict(Counter(r["evaluation_truth"]["kind"] for r in original)),
        variants=dict(Counter(r["variant"] for r in rows)),verdict_label_sources=dict(Counter(r["target_sources"]["verdict"] for r in rows)),
        mixed_temporal_views=sum(any(c["predicate"]=="temporal_pattern_difference" for c in claim_catalog(r["input"])) for r in original),
        supervised_train_rows={f:sum(bool(r["supervision"].get(f)) for r in rows if r["split"]=="train") for f in ("claims","verdict","explanation")},
        human_verdict_gold_rows=sum(r["evaluation_truth"].get("verdict_source")=="human_reviewed" and r["evaluation_truth"].get("verdict") is not None for r in rows))


def review_view(dataset,out):
    """Readable per-case files; authoritative edits still use review JSONL."""
    rows=read_jsonl(dataset);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    index=["# Qwen 근거·학습 초안 검수", "", "자동 초안은 사람 정답이 아닙니다. review.jsonl에서 검수자·정답 출처·마스킹을 수정하고 review-apply로 적용하세요.", "",
        "| 사례 | 분할 | 원곡 그룹 | 종류 | 변형 | 검수 상태 |", "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        name="case_"+digest(row["case_id"])[:16]+".md"
        index.append(f"| [{row['case_id']}]({name}) | {row['split']} | {row['source_group_id']} | {row['evaluation_truth']['kind']} | {row['variant']} | {row['review']['status']} |")
        body=[f"# {row['case_id']}","",f"분할: {row['split']} / 원곡 그룹: {row['source_group_id']}","",
            "## 검수·출처·손실 마스크","","```json",json.dumps({k:row[k] for k in ('review','target_sources','supervision','original_evidence')},ensure_ascii=False,indent=2),"```","",
            "## 자동 출력 초안","","```json",json.dumps(row['target'],ensure_ascii=False,indent=2),"```","",
            "## 실제 입력 근거","","```json",json.dumps(row['input'],ensure_ascii=False,indent=2),"```"]
        (out/name).write_text("\n".join(body)+"\n")
    (out/"index.md").write_text("\n".join(index)+"\n");review_export(dataset,out/"review.jsonl")
    (out/"coverage.json").write_text(json.dumps(dataset_coverage(rows),ensure_ascii=False,indent=2))
    return dict(index=str(out/"index.md"),review_file=str(out/"review.jsonl"),cases=len(rows))


def intervention_cases(payload):
    """Only label removal as necessary when there is no alternative support."""
    result=[("original",copy.deepcopy(payload),dict(kind="baseline", expected="same"))]
    swap=copy.deepcopy(payload)
    for e in swap["evidence"]:
        e["a"],e["b"]=e["b"],e["a"]
        for key in ("ordered_music","structure_context"):
            e[key]["a"],e[key]["b"]=e[key]["b"],e[key]["a"]
        context=e["structure_context"]
        if "a_status" in context: context["a_status"],context["b_status"]=context["b_status"],context["a_status"]
        for w in e["alignment"]["windows"]: w["a"],w["b"]=w["b"],w["a"]
        duration=e["alignment"]["duration_seconds"];duration["a"],duration["b"]=duration["b"],duration["a"]
    result.append(("swap_ab",swap,dict(kind="invariance",expected="same")))
    reverse=copy.deepcopy(payload);reverse["evidence"].reverse()
    result.append(("reverse_evidence",reverse,dict(kind="invariance",expected="same")))
    # A necessary-evidence test retains one source of musical support first.
    # This avoids incorrectly demanding a verdict flip with redundant matches.
    core=copy.deepcopy(payload);core["evidence"]=core["evidence"][:1]
    result.append(("unique_core",core,dict(kind="baseline",expected="same")))
    removed=copy.deepcopy(core);removed["evidence"]=[]
    if "retrieval" in removed:removed["retrieval"]=dict(status="evidence_removed_by_intervention",matrix_maxima={},threshold=None)
    result.append(("remove_unique_core",removed,dict(kind="necessity",expected="insufficient_evidence",reference_variant="unique_core",redundant=False)))
    irrelevant=copy.deepcopy(payload)
    if irrelevant["evidence"]:
        item=copy.deepcopy(irrelevant["evidence"][0]);item["evidence_id"]="E999";item["aliases"]=[]
        for f in item["factors"].values():
            f.update(status="unavailable",raw_cosine=None,provenance=dict(status="missing",kinds=[]))
            f["persistence"]=dict(status="unavailable",n_windows=0,above_threshold=0,mismatches=[])
        item["ordered_music"]={"a":[],"b":[]};item["temporal_comparison"]={}
        item["structure_context"]={"a":[],"b":[],"role":"context_only_not_similarity_evidence"}
        irrelevant["evidence"].append(item)
    result.append(("add_irrelevant",irrelevant,dict(kind="invariance",expected="same",irrelevant_evidence_ids=["E999"])))
    low=copy.deepcopy(payload)
    if "retrieval" in low:low["retrieval"]=dict(status="information_masked_by_intervention",matrix_maxima={},threshold=None)
    for e in low["evidence"]:
        for f in e["factors"].values():
            f["analyzer_reliability"]=dict(status="not_validated",n=0)
            f["status"]="unavailable";f["raw_cosine"]=None;f["provenance"]=dict(status="missing",kinds=[])
        e["ordered_music"]={"a":[],"b":[]};e["temporal_comparison"]={}
    result.append(("missing_information",low,dict(kind="availability",expected="insufficient_evidence")))
    return result


def prepare_pilot(manifest_path,out,backend,max_per_split=3,evidence_limit=2,time_bins=6,reports=None,supervise_rule_verdict=False):
    from .compare import analyze_song,compare_songs,public_song
    from .judge import evidence_payload
    from .train import audit_manifest
    manifest=json.loads(Path(manifest_path).read_text());audit_manifest(manifest)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Connected components across ALL manifest pairs, before any sampling or
    # intervention. Derived examples never get a new independently split ID.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    def join(ids):
        for x in ids[1:]: parent[find(x)]=find(ids[0])
        if ids: find(ids[0])
    cases=[]
    for p in manifest["pairs"]:
        join(p["original_ids"])
    for p in manifest.get("insertion_cases",[]): join(p["original_ids"])
    group_members={}
    for x in list(parent):group_members.setdefault(find(x),[]).append(x)
    groups={x:"sg_"+digest(sorted(group_members[find(x)]))[:12] for x in parent}
    for split in ("train","validation","test"):
        selected=[p for p in manifest["pairs"] if p["split"]==split]
        # First one per kind, then fill. A small pilot shouldn't accidentally
        # contain only transpositions because of manifest order.
        kinds=list(dict.fromkeys(p["kind"] for p in selected))
        kinds.sort(key=lambda k:(k!="hard_negative",k!="transpose",k))
        chosen=[next(p for p in selected if p["kind"]==k) for k in kinds]
        chosen+=(p for p in selected if p not in chosen)
        for p in chosen[:max_per_split] if max_per_split else chosen:
            cases.append(dict(a=p["a"]["audio"],b=p["b"]["audio"],pair=p))
    for p in manifest.get("insertion_cases",[]):cases.append(dict(a=p["a"],b=p["b"],pair=p))
    cache,rows={},[]
    for n,case in enumerate(cases):
        p=case["pair"]
        songs={}
        for side in ("a","b"):
            path=case[side]
            if path not in cache:cache[path]=analyze_song(path,backend)
            songs[side]=cache[path]
        comparison=compare_songs(songs["a"],songs["b"],backend,threshold=-1.,top_k=12)
        report=dict(schema_version=1,backend=backend.metadata(),songs={s:public_song(v) for s,v in songs.items()},comparison=comparison)
        report_path=out/"pipeline_reports"/f"case_{n:03}.json";report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(report,ensure_ascii=False))
        payload=compact_evidence(comparison,backend.metadata(),songs,limit=evidence_limit,time_bins=time_bins)
        legacy=evidence_payload(comparison["evidence"])
        base_id="case_"+digest(dict(ids=p["original_ids"],a=case["a"],b=case["b"],kind=p["kind"]))[:12]
        for variant,inp,intervention in intervention_cases(payload):
            target=draft_response(inp)
            validation=validate_claim_response(target,inp)
            if not validation["valid"]:raise ValueError(validation)
            # Legacy A sees the same intervention at evidence-level. Its
            # summaries cannot represent changes within temporal sequences.
            legacy_variant=copy.deepcopy(legacy)
            if variant in ("unique_core","remove_unique_core"):legacy_variant=legacy_variant[:1] if variant=="unique_core" else []
            elif variant=="swap_ab":
                for e in legacy_variant:
                    e["a"],e["b"]=e["b"],e["a"]
                    for key in ("music_information","structure_context"):e[key]["a"],e[key]["b"]=e[key]["b"],e[key]["a"]
            elif variant=="reverse_evidence":legacy_variant.reverse()
            elif variant=="missing_information":
                for e in legacy_variant:
                    for f in e["factors"].values():f.update(status="unavailable",cosine=None)
                    e["music_information"]={"a":[],"b":[]}
            elif variant=="add_irrelevant" and legacy_variant:
                e=copy.deepcopy(legacy_variant[0]);e["evidence_id"]="E999"
                for f in e["factors"].values():f.update(status="unavailable",cosine=None)
                e["music_information"]={"a":[],"b":[]};legacy_variant.append(e)
            availability_label=variant in ("remove_unique_core","missing_information")
            rows.append(dict(case_id=base_id+"_"+variant,base_case_id=base_id,variant=variant,
                source_group_id=groups[p["original_ids"][0]],original_ids=p["original_ids"],split=p["split"],
                input=inp,legacy_input=legacy_variant,target=target,supervision=dict(claims=True,verdict=availability_label or supervise_rule_verdict,explanation=True),
                review=dict(status="auto_generated",reviewer=None,notes="Needs human music/evidence review"),
                target_sources=dict(claims="deterministic_evidence_predicates",verdict="evidence_availability_definition" if availability_label else "rule_provisional",explanation="template_from_observable_evidence"),
                generation=dict(method="pipeline_prediction_then_rule_draft",backend=backend.metadata(),manifest_sha256=file_digest(manifest_path)),
                original_evidence=dict(report_path=str(report_path),report_sha256=file_digest(report_path)),
                evaluation_truth=dict(kind=p["kind"],factor_labels=p.get("targets",(p.get("truth") or {}).get("factors",{})),
                    truth=p.get("truth"),verdict=None,verdict_source="not_human_annotated",
                    controlled_pattern_reuse=p["kind"]=="insertion",annotation_role="review_and_evaluation_only"),
                intervention=intervention,semantic_review=dict(supported_claim_ids=None,unsupported_claim_ids=None,contradiction=None)))
    # Existing reports may only be imported with an explicit manifest mapping;
    # matching the original audio paths prevents inventing new source groups.
    if reports:
        index={(str(Path(c["a"]).resolve()),str(Path(c["b"]).resolve())):c for c in cases}
        for path in reports:
            r=json.loads(Path(path).read_text()); key=tuple(str(Path(r["songs"][s]["path"]).resolve()) for s in ("a","b"))
            if key not in index:raise ValueError("Report has no sampled manifest provenance; increase sample or provide matching manifest")
            p=index[key]["pair"];inp=compact_evidence(r["comparison"],r["backend"],r["songs"],limit=evidence_limit,time_bins=time_bins)
            template=next(copy.deepcopy(v) for v in rows if v["original_ids"]==p["original_ids"] and v["variant"]=="original")
            template.update(case_id="report_"+file_digest(path)[:16],base_case_id="report_"+file_digest(path)[:16],input=inp,
                            legacy_input=evidence_payload(r["comparison"]["evidence"]),target=draft_response(inp),
                            original_evidence=dict(report_path=str(path),report_sha256=file_digest(path)))
            template["generation"].update(method="existing_report_import_then_rule_draft",backend=r["backend"],
                                          temporal_policy="unavailable_when_original_frames_are_missing")
            rows.append(template)
    audit=audit_source_groups(rows);write_jsonl(out/"pilot.jsonl",rows)
    summary=dict(**audit,rows=len(rows),base_cases=len({r["base_case_id"] for r in rows}),
        review_counts={"auto_generated":len(rows),"human_approved":0},
        split_rows={s:sum(r["split"]==s for r in rows) for s in ("train","validation","test")},
        manifest_originals={s:sum(v==s for v in manifest.get("split_originals",{}).values()) for s in ("train","validation","test")},
        semantic_gold_count=0,note="Automatic pilot is optimization smoke data, not human-reviewed research evaluation")
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def review_export(dataset,out):
    rows=read_jsonl(dataset)
    write_jsonl(out,[dict(case_id=r["case_id"],review=r["review"],target=r["target"],supervision=r["supervision"],
        target_sources=r["target_sources"],evaluation_truth=r["evaluation_truth"],semantic_review=r["semantic_review"]) for r in rows])


def prepare_order_test(dataset,out):
    """A nontrivial held-out order test from existing frozen predictions."""
    rows=read_jsonl(dataset)
    base=next(r for r in rows if r["split"]=="test" and r["variant"]=="original" and r["evaluation_truth"]["kind"]=="insertion")
    path=base["original_evidence"]["report_path"]
    if file_digest(path)!=base["original_evidence"]["report_sha256"]:raise ValueError("Source prediction report changed")
    report=json.loads(Path(path).read_text())
    payload=compact_evidence(report["comparison"],report["backend"],report["songs"],limit=2,time_bins=4)
    if len(payload["evidence"])<2:raise ValueError("Need two actual representative matches, not duplicated fake evidence")
    result=[]
    for variant,inp,label in intervention_cases(payload):
        if variant not in ("original","reverse_evidence"):continue
        row=copy.deepcopy(base);row["base_case_id"]=base["base_case_id"]+"_order2";row["case_id"]=row["base_case_id"]+"_"+variant
        row.update(variant=variant,input=inp,target=draft_response(inp),intervention=label,
                   review=dict(status="auto_generated",reviewer=None,notes="Frozen report, two distinct candidates; not human gold"))
        if variant=="reverse_evidence":row["legacy_input"].reverse()
        row.pop("soft_cache",None);result.append(row)
    audit_source_groups(result);write_jsonl(out,result)
    return dict(rows=len(result),evidence_per_case=2,source_group_id=base["source_group_id"],split="test",source_report=path)


def review_apply(dataset,patches,out):
    rows=read_jsonl(dataset);byid={r["case_id"]:r for r in rows};seen=set()
    for patch in read_jsonl(patches):
        cid=patch["case_id"]
        if cid not in byid or cid in seen:raise ValueError("Unknown/duplicate review case_id")
        seen.add(cid);row=byid[cid]
        if patch["review"]["status"]=="human_approved" and not patch["review"].get("reviewer"):
            raise ValueError("Human approval needs an explicit reviewer")
        for key in ("review","target","supervision","target_sources","evaluation_truth","semantic_review"):
            if key in patch:row[key]=patch[key]
        if row["review"]["status"]=="human_approved" and row["supervision"].get("verdict") and row["target_sources"]["verdict"] not in ("human_reviewed","controlled_pattern_reuse","evidence_availability_definition"):
            raise ValueError("Supervising an approved verdict requires its explicit reviewed label source")
        check=draft_response(row["input"])
        for key,value in row["target"].items():
            if value is not None:check[key]=value
        validation=validate_claim_response(check,row["input"])
        if not validation["valid"]:raise ValueError(validation)
    audit_source_groups(rows);write_jsonl(out,rows)

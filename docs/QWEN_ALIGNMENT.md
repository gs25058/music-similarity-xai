# Qwen 근거 해석·주장 검증·LoRA 실험

기준일: **2026-11-05**. 기존 MVP를 이어서 확장했다. 기존 명령과 음악 체크포인트는 유지하며, 새 경로는 `--qwen-config`로 선택한다. 수치와 산출물은 [VALIDATION.md](VALIDATION.md)에 기록한다.

현재 확인한 개선은 **자동 파일럿에서 근거 계약에 맞는 주장·판단을 생성하고, 근거가 없을 때 보류하는 동작**이다. 사람이 검수한 음악적 판단·자연어 충실도 정답은 아직 없으며, 그 정확도가 개선됐다는 결과는 아니다.

## 근거 입력과 출처

`src/evidence.py`의 `compact_evidence`가 `compare_songs`의 실제 예측을 변환한다. 보고서 원본은 보존한다.

| 정보 | 처리·한계 |
| --- | --- |
| 대응 구간 | 원본 `evidence_id`, A/B 시간, 창별 정렬과 지속성. 음표 단위 정렬로 해석하지 않는다. |
| 요소 점수 | 코사인 원값·방향·범위와 정의. 확률이 아니다. 헤드·체크포인트·백본 리비전을 식별한다. 새 음악 추론에는 체크포인트 SHA-256도 기록한다. |
| 분석 신뢰도 | 별도의 `analyzer_reliability`. 현재 정확도 정답이 없어 `not_validated`이며 높은 점수로 대체하지 않는다. |
| 시간순 음악 정보 | 주선율·저음 음고와 상대 변화, 온셋과 상대 간격, 시간순 크로마·가용 코드 예측. 연속 시간 bin 요약이다. |
| 박 기준 온셋 | 박 예측에서 충분한 peak가 있을 때만 산출한다. 검증된 박 정답이 아니며 없으면 `unavailable`이다. |
| 누락·불일치 | 학습하지 않은 예측, 유성 구간 부족, 낮은 대응 창, 시간 요약의 차이를 별도로 표시한다. |
| 구조 | SongFormer 기능 라벨·경계는 맥락이다. 같은 후렴이라는 이유로 점수를 더하지 않는다. |

시간 정보를 평균·최빈값 하나로 대체하지 않는다. 원본 창별 정보를 시간순으로 유지하고, 입력 bin 수를 줄일 때도 양끝을 포함한 순서 표본임을 기록한다. 상대 음정 변화는 **인접한 원래 bin 사이의 차이**다. 표본에서 앞 bin이 생략될 수 있어 최신 형식에는 기준 bin의 시간·음고와 원래 bin 인덱스를 함께 넣는다. 이전 파일럿의 상대 변화도 원래 bin 기준이며, 남은 표본끼리의 차이로 읽으면 안 된다.

DSP는 혼합 음원의 파생 프록시, 학습 헤드는 검증되지 않은 예측으로 표시한다. 피아노 저음 파생 정답과 실제 베이스 MIDI의 학습 출처도 구분한다. 과거 보고서에 프레임 정보가 없으면 시간순 음표를 복원해 만들지 않고 기존 창 요약과 누락 상태를 보존한다.

`representative_evidence`는 양쪽 구간 IoU를 사용하는 NMS로 겹치는 후보를 줄이고 원래 ID와 제외한 ID 별칭을 보존한다. 기본 IoU는 0.6이다. 부분적으로 겹치는 모든 후보가 독립이라는 뜻은 아니다.

### 점수 보정

```bash
.venv/bin/python scripts/qwen_dataset.py calibrate \
  --manifest data/synthetic/manifest.json --backend ours \
  --checkpoint checkpoints/gpu/mert95_smoke.pt --device cuda \
  --out results/qwen/calibration.json
```

validation의 시간 정렬 창에서 balanced accuracy 기준 임계값을 선택한다. test 보정은 거부한다. 유효 창 수·양성/음성 수·원곡 ID·정답 적용 범위·manifest 및 모델 식별자를 기록한다. 정답은 통제된 변형 쌍의 요소 라벨을 정렬 창에 적용한 것이며 사람 정답이 아니다.

이는 **임계값 보정**이며 확률 보정이 아니다. 분석기의 정확도를 검증하지도 않는다. 창 쌍에서 맞춘 임계값을 긴 연결 후보의 평균에 그대로 적용하지 않도록 점수 단위를 검사한다. 불일치하면 해당 후보는 `uncalibrated`로 남긴다. `config`의 `calibration`에 파일 경로를 지정해 적용한다. 파일이 없을 때 기본 0.8은 미보정 해석 기준이다. 후보 검색 임계값과 요소별 해석 임계값도 구분한다.

## 주장 출력과 검증 범위

새 출력은 다음 형태다. `assertion`은 수치·시간을 주장할 때 사용하는 선택 필드다.

```json
{
  "claims": [
    {"claim_id":"C001","evidence_ids":["E001"],"factor":"rhythm","role":"supports","predicate":"score_high","statement":"리듬의 원시 유사도가 설정된 해석 기준 이상입니다."},
    {"claim_id":"C002","evidence_ids":["E001"],"factor":"rhythm","role":"limits","predicate":"uncalibrated_score","statement":"리듬 점수는 보정되지 않아 판단의 확실성으로 해석할 수 없습니다."}
  ],
  "verdict":"review",
  "explanation":"대응 구간의 리듬 점수가 높게 관찰됩니다. 점수 보정과 분석 정확도의 한계를 함께 검토해야 합니다."
}
```

`src/claims.py`가 입력 근거로부터 허용할 수 있는 주장 목록을 만든다. 모델은 실제 사용하는 주장과 문장을 생성한다.

| predicate | 검사하는 사실 |
| --- | --- |
| `score_high`, `score_low` | 해당 요소 점수와 해석 기준의 비교 방향 |
| `observed_value` | `assertion`의 코사인·시간 값과 `eq/ge/le` 관계 |
| `uncalibrated_score`, `analyzer_unvalidated` | 보정 및 분석 정확도 검증의 상태 |
| `information_missing` | 해당 요소의 누락 상태 |
| `persistent_match`, `conflicting_windows` | 여러 대응 창에서 기준 이상 여부와 불일치 |
| `temporal_pattern_match`, `temporal_pattern_difference` | 시간 요약의 근사 비교. 원시 임베딩 점수와 별개의 미보정 파생 비교다. |
| `same_section_context`, `different_section_context` | 기능 라벨의 맥락. 내용 유사성을 지지하는 역할로 사용할 수 없다. |

자동 검사는 알려진 ID·요소·출처·가용성, predicate/role 조합, 수치·시간·비교 방향, 근거 없는 숫자, 구조와 내용의 혼동, 일부 명시적인 판단·설명 모순을 확인한다. 높은 판단에는 여러 요소의 지지가 필요하며, 미검증 근거의 제약 주장도 요구한다.

여러 요소 지지를 요구하는 것은 현재의 보수적 실험 정책이며 인간 지각의 보편 법칙은 아니다. 화성이 비슷해도 음정 진행 등 반대 근거가 있으면 낮은 종합 판단을 허용한다. 사람 정답을 확보한 뒤 판단 정책 자체도 평가해야 한다.

**자유 문장의 모든 의미를 자동 판정하지 않는다.** predicate가 맞아도 문장 안에 새로운 음악 사실을 넣거나 미묘한 모순을 만들 수 있다. `human_review_required`와 `semantic_accuracy`로 이 범위를 구분한다. 자동 주장 지지율은 이 계약 검사에 대한 비율이다.

보고서의 `judgment.selected.judgment/evidence_ids/explanation/cited_intervals`는 기존 소비자를 위해 유지한다. 추가 필드는 `claim_response`, `claim_validation`, `model_validation`, `generation_trace`, `evidence_input`이다. 시간은 원본 근거에서 표시한다. 원본 응답·시도별 검증·수정 응답을 보존한다.

기존 `rule_baseline`도 보존한다. Qwen 실패 시 기존 규칙을 주장 계약으로 표현할 수 있으면 사용하고, 그렇지 않으면 검증 가능한 보수적 주장 규칙을 사용한다. 실제 선택은 `execution.fallback_source`, 사용 여부는 `execution.fallback_used`에 기록한다. 부모 프로세스도 worker 응답을 독립 검증하며 `failed_rule_fallback`은 Qwen 성공에 포함하지 않는다. 별도 품질 비교기는 fallback을 실행하지 않고 `fallback_required_rate`와 `fallback_used_rate`를 구분한다. 기존 `fallback_rate`는 필요율의 호환 별칭이다.

## 데이터 준비·검수

핵심 음악 모델은 고정한 채 실제 파이프라인 예측에서 파일럿을 만든다. MIDI 정답은 `evaluation_truth`에 따로 보관하며 설명 초안은 입력에서 관측할 수 있는 사실로 작성한다.

```bash
.venv/bin/python scripts/qwen_dataset.py prepare \
  --manifest data/synthetic/manifest.json --backend ours \
  --checkpoint checkpoints/gpu/mert95_smoke.pt --device cuda \
  --max-per-split 2 --evidence-limit 1 --time-bins 4 --out data/qwen_pilot
.venv/bin/python scripts/qwen_dataset.py audit --dataset data/qwen_pilot/pilot.jsonl
.venv/bin/python scripts/qwen_dataset.py review-view \
  --dataset data/qwen_pilot/pilot.jsonl --out data/qwen_review
```

`review-view`는 읽기용 사례별 Markdown과 `index.md`, 수정용 `review.jsonl`을 만든다. 다음을 검수한다.

1. 입력·출처·누락 상태와 주장·설명·판단이 맞는지 확인한다.
2. 확인한 필드만 `supervision`을 켠다. 미확정 필드는 끄거나 `null`로 둔다.
3. 판단 라벨의 출처를 `target_sources.verdict`에 명시한다. 사람 판단, 통제 재사용, 규칙 임시 라벨을 혼동하지 않는다.
4. 승인할 때 `review.status=human_approved`, 실제 검수자를 `review.reviewer`에 기록한다.

```bash
.venv/bin/python scripts/qwen_dataset.py review-apply \
  --dataset data/qwen_pilot/pilot.jsonl --patches data/qwen_review/review.jsonl \
  --out data/qwen_pilot/reviewed.jsonl
```

원곡 ID와 공동 사례의 연결 성분으로 source group을 만든 뒤 기존 원곡 분할을 따른다. 변형·설명·근거 수정은 같은 분할에 둔다. 보고서 import는 manifest의 실제 음원 경로에 매핑되는 경우만 허용한다. 원곡이나 그룹이 분할을 넘으면 오류다.

실행한 작은 파일럿은 **64행, 원음 쌍 9개와 과거 보고서 view 1개, 사용 원곡 ID 6개, source group 3개**다. train/validation/test는 각 한 그룹이다. 원래 합성 manifest에는 원곡 12개가 있고 8/2/2로 나뉜다. `--max-per-split 0`으로 표본 제한을 해제할 수 있지만 증강 행 수를 독립 원곡 수로 해석하면 안 된다.

원래 근거, A/B 교환, 순서 변경, 유일 근거만 유지, 그 근거 제거, 무관한 누락 근거 추가, 정보 누락을 포함한다. 원음 쌍은 어려운 음성 쌍·조옮김·삽입 사례다. 시간 요약과 원시 점수가 엇갈리는 view도 있다. 모두 `auto_generated`이며 사람 검수는 **0건**이다.

### 판단 정답과 마스킹

기본 준비 명령은 미검수 규칙 판단을 마스킹한다. 근거가 완전히 제거되거나 모든 음악 정보가 누락된 사례의 보류 판단만 `evidence_availability_definition` 출처로 학습한다. 그 외 종합 판단 정답은 확보하지 않았다.

자동 정책을 배우는 smoke는 `prepare --supervise-rule-verdict`와 `configs/qwen/lora_weak_policy_smoke.json`으로 **명시적으로** 선택한다. 이는 `rule_provisional` 라벨 학습이며 사람의 음악적 유사성 정답이 아니다. 현재 실험의 `pilot_weak_policy.jsonl`은 검수 patch 도구로 그 설정을 켠 별도 복사본이다. 기본 학습 코드는 명시적 허용 없이 임시 판단을 학습하는 것을 거부한다.

## LoRA 학습·재개·추론

설치는 기존 Qwen 별도 환경에서 `pip install -r requirements/qwen.txt`로 갱신한다. PEFT 0.21.2, Accelerate 1.15.0을 사용했다. 공식 Qwen3.5-0.8B의 실제 모듈을 확인해 다음에 rank 16 LoRA를 적용했다.

```text
model.language_model.layers.*.self_attn.q_proj / v_proj
model.language_model.layers.*.linear_attn.in_proj_qkv / out_proj
```

학습 파라미터는 **3,588,096개**다. 음악 인코더·요소 헤드·SongFormer와 Qwen 원본 가중치·vision 분기는 고정한다. LoRA와 선택적 연결층만 학습한다. 모델을 축소하거나 양자화하지 않았다.

```bash
# 자동 데이터의 최적화·저장 smoke. 기본 임시 판단은 마스킹한다.
.venv-qwen/bin/python scripts/train_qwen.py --dataset data/qwen_pilot/pilot.jsonl \
  --config configs/qwen/lora_smoke.json --max-steps 16 \
  --out checkpoints/qwen/lora_smoke --device cuda

# 검수 자료 확보 후 본 학습. 기본 설정은 human_approved만 선택한다.
.venv-qwen/bin/python scripts/train_qwen.py --dataset data/qwen_pilot/reviewed.jsonl \
  --config configs/qwen/lora.json --out checkpoints/qwen/reviewed --device cuda

# 같은 데이터·설정에서 optimizer와 RNG 상태를 불러와 이어서 학습한다.
.venv-qwen/bin/python scripts/train_qwen.py --dataset data/qwen_pilot/pilot.jsonl \
  --config configs/qwen/lora_smoke.json --resume checkpoints/qwen/lora_smoke \
  --max-steps 17 --out checkpoints/qwen/lora_resumed --device cuda
```

입력·padding·미확정 출력은 `-100`으로 마스킹한다. 판단, 인용, 주장, 설명의 토큰 수·평균 CE·가중 손실 합을 각각 기록한다. 모든 학습 사례의 길이를 최적화 전에 검사하며 정답을 조용히 잘라내지 않는다. 더 긴 근거를 쓸 때는 bin/후보 설정과 config의 `max_length`를 명시적으로 맞춘다.

기본 설정은 batch 1, 출력 토큰의 가중 CE다. `logits_to_keep`로 필요한 출력 위치만 vocabulary projection한다. seed, 모델 리비전, 데이터 SHA, source group, 학습 파라미터와 모듈, 실제 사용 사례, 손실 이력을 어댑터와 함께 저장한다. 소규모 실험은 마지막 step을 저장했으며 validation으로 최적 checkpoint를 선택한 실험은 아니다.

현재 기본 길이는 12,288 token이며 최신 형식의 8,832 token 사례로 실제 최적화를 확인했다. 과거 실험의 정확한 재현에는 당시 JSONL과 checkpoint의 SHA가 필요하다. 현재 준비 코드로 다시 생성하면 추가된 bin 참조·출처 등으로 토큰 수와 손실 수치가 달라질 수 있다.

추론 config의 `adapter`에 어댑터 폴더를 지정한다. 기존 worker도 stdin request의 선택적 `adapter`를 지원한다. 어댑터의 base revision과 현재 Qwen revision이 다르면 실패한다.

```bash
python -m src compare song_a.wav song_b.wav --backend ours --checkpoint checkpoints/mert95_smoke.pt \
  --judge qwen --qwen-python .venv-qwen/bin/python --qwen-config configs/qwen/inference.json \
  --songformer-python .venv-songformer/bin/python --device cuda --out results/claims.json
```

위 기본 config는 새 근거·주장 프로토콜의 고정 Qwen이다. LoRA를 쓰려면 그 config를 복사해 `adapter`를 추가한다. `--qwen-config`를 생략하면 기존 요약·판단 출력 경로다.

## 고정 근거 충실도 평가와 사람 의미 평가

```bash
.venv/bin/python scripts/evaluate_qwen.py --dataset data/qwen_pilot/pilot.jsonl \
  --adapter checkpoints/qwen/lora_smoke --conditions A B C \
  --config configs/qwen/inference_smoke.json --base-cases 1 --device cuda \
  --out results/qwen/abc.json
```

A는 기존 요약과 기존 출력, B는 개선 근거와 주장 출력의 고정 Qwen, C는 B와 같은 입력·디코딩의 LoRA다. 따라서 A/B 차이는 입력만의 효과로 해석하지 않는다. B/C는 입력·실제 렌더링한 prompt·디코딩·구현 해시를 확인한다.

판단 정확도와 macro-F1은 출처가 있는 판단 정답이 있을 때 계산한다. 사람 정답이 없으면 `null`이다. 자동 주장 지지율, 잘못된 인용·지원 범위, 일부 모순, 근거 제거·무관 추가·A/B 교환·순서 변경, JSON 실패·재시도·fallback, 토큰 수·모델 로드·캐시·입력 처리·생성 시간을 기록한다. 실패 후 규칙 출력을 모델 성공으로 세지 않는다.

후보가 하나인 순서 교환은 `trivial_identical_input_n`으로 제외한다. 실제 두 후보 순서 검사는 저장된 예측으로 만든다.

```bash
.venv/bin/python scripts/qwen_dataset.py order-test --dataset data/qwen_pilot/pilot.jsonl --out data/qwen_order/pilot.jsonl
.venv/bin/python scripts/evaluate_qwen.py --dataset data/qwen_order/pilot.jsonl \
  --adapter checkpoints/qwen/lora_smoke --conditions B C --base-cases 1 --device cuda \
  --out results/qwen/order.json
```

유일 근거 제거는 먼저 `unique_core`로 대체 근거를 없앤 조건을 만든다. 여러 대응이 남은 일반적인 제거와 혼동하지 않는다. 인용 ID 집합 일치율도 기록하지만, 다른 유효 근거를 골랐다는 사실만으로 의미적 오류라고 판단하지 않는다.

결과 옆의 `.semantic_review.jsonl`에서 **정확한 원본 응답 hash**, 검수자, 지지/비지지 claim ID, 모순 여부를 기록한다. A의 자유 설명에는 `statement_spans`로 의미를 평가할 수 있다. 빈 검수 양식은 평가 완료로 세지 않는다.

```bash
.venv/bin/python scripts/score_qwen_outputs.py --report results/qwen/abc.json \
  --dataset data/qwen_pilot/reviewed.jsonl --reviews results/qwen/abc.json.semantic_review.jsonl \
  --out results/qwen/abc_reviewed.json
```

이 명령은 출력을 다시 생성하지 않는다. 입력이 그대로인지와 응답 hash·인용 claim ID를 검사한 후 사람 주석을 집계한다. 반례 문장을 찾는 자동 규칙의 결과를 사람 의미 정확도로 바꾸지 않는다.

### 오디오 자체를 바꾼 별도 평가

```bash
.venv/bin/python scripts/evaluate_audio_interventions.py --manifest data/synthetic/manifest.json \
  --backend ours --checkpoint checkpoints/gpu/mert95_smoke.pt --config /path/to/adapter_inference.json \
  --songformer-python .venv-songformer/bin/python --device cuda --out results/qwen/audio_reanalysis
```

알려진 B 삽입 구간만 새 WAV에서 무음으로 만들고 설정한 음악 분석기·SongFormer·Qwen을 다시 실행한다. 원음은 보존한다. `evaluation_mode=audio_reanalysis`로 고정 예측의 근거 제거와 분리한다. 절단 경계 효과와 다른 유사 구간의 잔존 가능성이 있어 전역 판단이 반드시 바뀐다고 라벨링하지 않는다.

## 디코더 설정과 회귀 검사

`decoding.json_constraints`, `prose_filter`, `forbid_numeric_prose`, `repetition_penalty`, `no_repeat_ngram`, `max_retries`를 분리했다. B/C는 같은 설정을 받는다. 새 경로의 기본 반복 제한은 꺼져 있다. 여러 주장에 필요한 반복 구문까지 막는 영향을 비교하기 위해서다. 기존 경로 설정은 유지한다.

실제 토크나이저에서 한글·구두점·숫자 assertion·근거 ID·UTF-8·빈 근거를 검사한다. LMFE가 놓치던 `.",` 형태의 문자열 경계 토큰을 작은 후보 집합의 parser 검사로 허용한다. Unicode 조건은 LMFE가 덮어쓴 문자 범위도 복원한다.

토크나이저의 `im_end` **248046**과 모델 기본 종료 **248044**가 달라, 이전에는 JSON 이후에도 종료 토큰을 계속 생성했다. 두 worker가 tokenizer 종료 ID를 명시하도록 수정했다.

```bash
.venv-qwen/bin/python -m unittest discover -s tests -p test_qwen_decoding.py -v
.venv/bin/python scripts/evaluate_qwen_decoding.py --dataset data/qwen_pilot/pilot.jsonl \
  --adapter checkpoints/qwen/lora_smoke --device cuda --out results/qwen/decoding
```

두 번째 명령은 같은 원본 사례에서 기본 설정, JSON 제약 해제, Unicode 범위, 반복 제한을 하나씩 바꾼다. 사람 의미 정답 없는 한 사례의 엔지니어링 검사다.

## 선택적 음악 soft token

```bash
.venv/bin/python scripts/prepare_soft_cache.py --dataset data/qwen_pilot/pilot.jsonl \
  --checkpoint checkpoints/gpu/mert95_smoke.pt --device cuda --out data/qwen_soft
.venv-qwen/bin/python scripts/train_qwen.py --dataset data/qwen_soft/pilot_with_soft.jsonl \
  --config configs/qwen/lora_smoke.json --soft-config configs/qwen/soft_tokens.json \
  --max-steps 16 --out checkpoints/qwen/soft_smoke --device cuda
.venv/bin/python scripts/evaluate_qwen.py --dataset data/qwen_soft/pilot_with_soft.jsonl \
  --adapter checkpoints/qwen/lora_smoke --soft-adapter checkpoints/qwen/soft_smoke \
  --conditions C D --config configs/qwen/inference_smoke.json --base-cases 1 --device cuda \
  --out results/qwen/cd.json
```

고정한 네 헤드의 시간축 64차원 특징으로 A, B, 차이, 절대 차이, 요소 one-hot, 양쪽 시간 위치를 연결한 **262차원** 행을 만든다. 작은 projection과 순서 resampling으로 기본 **16개**, Qwen 임베딩 차원 **1024**의 prefix를 만든다. 연결층은 **166,284개** 파라미터다. 명시적 근거와 ID를 함께 입력한다.

캐시는 원음 SHA, 백본 리비전, 음악 체크포인트 SHA, 24 kHz/20 Hz/패딩과 표본 설정, 실제 근거 fingerprint를 포함한다. worker에서 검증 후 Qwen 임베딩을 구성한다. prefix label은 마스킹하고 attention·padding·position을 맞춘다. 음악 근거가 없는 사례는 prefix attention도 꺼서 제거한 근거가 캐시로 남지 않게 한다.

같은 데이터·seed·step 수의 C/D를 비교하고, D의 prefix 제거·A/B 특징 교환을 추가 기록한다. 제거로 문맥 길이도 바뀌므로 반응 변화가 음악 정보 사용의 증명은 아니다. 동일 복사 패턴 사례의 A/B 교환도 다른 음악 내용을 넣은 교환 실험과 구분해야 한다. **soft token의 의미적 기여와 설명 가능 범위를 입증한 상태는 아니며 기본값은 비활성화다.**

주장 계약은 명시적 근거로 확인 가능한 사실만 허용한다. soft token만을 근거로 새로운 음표·코드 사실을 주장할 수는 없다. 잠재 특징이 판단을 바꿨다는 관찰과 그 변화를 자연어 근거로 충실하게 설명했다는 주장은 별도다.

## 참고와 다음 연구

[MusiLingo](https://aclanthology.org/2024.findings-naacl.231/)의 고정 MERT와 언어모델 사이 projection 아이디어를 참고했다. 이 구현은 대응하는 두 구간의 요소 특징·차이·시간 위치와 명시적 근거를 다루며 Qwen LoRA도 학습한다. 논문 코드 복사나 재현 성능을 주장하지 않는다.

[SelfCite](https://proceedings.mlr.press/v267/chuang25a.html)의 맥락 제거·유지 관점을 근거의 필요성·대체 가능성 검사에 적용했다. [공식 코드](https://github.com/facebookresearch/SelfCite)의 학습 코드는 가져오지 않았다. 자체 reward·선호 최적화는 아직 실행하지 않았다. 이후 선호 사례는 같은 source group·동일 입력의 chosen/rejected 응답, 양쪽 출력 hash, 검수자·비교 이유를 보존하고 train 분할에서만 만들어야 한다. test 출력의 선호를 학습에 재사용하면 안 된다.

[LoRA](https://arxiv.org/abs/2106.09685)는 PEFT 공개 API로 사용했다. 원본·버전·라이선스와 새 구현의 범위는 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)에 있다.

다음 핵심 작업은 더 많은 독립 원곡, 실제 혼합 음원·요소 불일치와 다양한 판단 라벨의 사람 검수다. 검수된 판단·설명 성능, 미묘한 의미 모순, 긴 곡·많은 후보, 보정의 일반화, soft token의 내용 의존성은 남은 검증이다.

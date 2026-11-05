# 실행 검증 기록

확인일: **2026-11-04~05, Asia/Seoul**. 기본 샌드박스에서는 GPU 장치가 보이지 않아 먼저 CPU 기준 검증을 수행했다. 이후 권한 확장 실행에서 RTX A6000 48GB GPU를 확인해 추가 소규모 검증을 수행했다. MERIT 330M, 우리 백본 MERT 95M, SongFormer, Qwen3.5-0.8B의 크기를 유지했다. 환경 버전은 [environment_versions.json](environment_versions.json)에 기록했다.

## 최신 단계: Qwen 근거·주장·LoRA·충실도 평가

구현과 사용법은 [QWEN_ALIGNMENT.md](QWEN_ALIGNMENT.md)에 있다. 기존 코드·현황·실제 보고서를 확인한 후 수정했다. RTX A6000에서 공식 Qwen 원본 가중치와 고정 음악 모델을 유지하며 실행했다. **기본 테스트 33개 통과, 선택 의존성 테스트 9개 건너뜀**을 확인했고, 별도 Qwen 환경에서 **실제 토크나이저 회귀 검사 9개를 모두 통과**했다.

### 데이터·학습·재로딩

`data/qwen_pilot_v2/pilot.jsonl`: **64행**, 원음 쌍 **9개 + 과거 보고서 view 1개**, 사용 원곡 ID **6개**, source group **3개**다. 그룹은 train/validation/test 각 하나이며 행 수는 21/21/22다. 원래 manifest에는 원곡 12개(train 8 / validation 2 / test 2)가 있다. 어려운 음성 쌍·조옮김·삽입과 입력 근거의 개입 사례를 만들었다. 원본 view 중 시간 요약의 차이가 관찰되는 view는 2개다.

모두 **자동 생성, 사람 검수 0건**이다. 기본 파일럿 train 21행에서 주장은 21행, 설명은 21행, 판단은 근거 제거·누락을 정의한 6행만 감독한다. 나머지 15행의 임시 판단은 마스킹한다. 별도 `pilot_weak_policy.jsonl`은 명시적으로 임시 정책 라벨 학습을 켠 실험이다. 이를 사람 정답으로 간주하지 않는다.

| 실행 | 실제 최적화 | 첫/마지막 손실 | 시간 | 결과 |
| --- | --- | --- | --- | --- |
| 마스킹 LoRA `lora_v2` | 16 step | 0.528747 / 0.002704 | 144.772초 | 저장, 새 worker 로드·추론 성공 |
| 마스킹 + prefix `soft_v2` | 16 step | 0.499135 / 0.005268 | 145.393초 | LoRA·연결층 저장·로드, 실제 gradient 확인 |
| 임시 정책 LoRA `lora_policy_smoke` | 16 step | 0.522975 / 0.003054 | 143.963초 | 저장, 새 worker 로드·추론 성공 |
| 같은 정책 + prefix `soft_policy_smoke` | 16 step | 0.508370 / 0.003372 | 143.586초 | 같은 데이터·seed·step의 C/D 조건 확인 |

각 LoRA는 **3,588,096개** 파라미터와 실제 attention 모듈을 기록했고, **96개 LoRA 텐서의 변경**을 확인했다. prefix 연결층은 별도 **166,284개** 파라미터다. 음악 모델은 고정했다. `lora_v2`의 저장된 optimizer·RNG·어댑터를 새 프로세스에서 읽어 **16 → 17 step** 재개도 실행했다. 첫 2 step smoke와 그 재개 기록도 로컬에 보존했다.

입력·패딩·미확정 필드·prefix 마스킹을 검사했다. BPE 토큰이 미확정 필드 경계를 걸치면 토큰 전체를 마스킹하며, 감독 가능한 정답이 하나도 없는 행은 학습에서 제외한다. 판단·인용·주장·설명별 토큰 수와 가중 손실 합을 이력에 기록했다. 명시적 근거가 없는 prefix 사례에는 gradient가 없는 것이 정상이고, 음악 특징이 있는 실제 학습에서는 연결층 gradient가 0보다 큼을 확인했다.

개발 중 무관 근거 추가 사례가 8,192 token을 넘는 것을 발견했다. 정답을 잘라내지 않고 중복 모델 메타데이터와 주장 목록 표현을 줄였다. 최종 준비 자료의 최대 학습 길이는 당시 **7,537 token**으로 사전 확인했다. 이후 추가한 출처·bin 참조를 포함해 새로 생성하는 자료는 다시 길이 검사를 통과해야 한다.

최신 형식에서는 같은 후보 수·bin 설정의 가장 긴 train 사례가 **8,832 token**이었다. 기본 config의 길이를 **12,288**로 늘리고 이 사례로 실제 **1 step 최적화·저장**을 실행했다(입력 8,589 / 출력 243 token, 23.375초, `long_input_probe`). 모델 크기는 유지했다. 이 한 사례는 길이·실행 경로 검사이며 성능 평가나 본 학습에 포함하지 않는다.

### 같은 예측 근거에서 A/B/C/D

파일: `results/qwen_alignment/abcd_final.json`, 재집계본 `abcd_scored.json`. **test 원곡 그룹 하나의 원음 쌍 한 개에 대한 7개 입력 view**다. 독립된 7곡 평가가 아니다. smoke의 재시도는 0회로 맞췄다.

| 조건 | 응답 검증 통과 | 자동 predicate 지지율 | 자동 모순 응답률 | fallback 필요율 | 평균 입력/출력 text token | 평균 생성 시간 |
| --- | --- | --- | --- | --- | --- | --- |
| A: 기존 입력·기존 출력·고정 Qwen | 7/7 | 해당 없음 | 해당 없음 | 0 | 2,830.9 / 152.1 | 19.076초 |
| B: 새 입력·주장 출력·고정 Qwen | 0/7 | 0.4091 | 0.1429 | 1 | 4,781.4 / 434.7 | 42.600초 |
| C: B + 임시 정책 LoRA | 7/7 | 1.0000 | 0 | 0 | 4,781.4 / 208.6 | 22.416초 |
| D: C와 같은 학습 조건 + 16 prefix | 7/7 | 1.0000 | 0 | 0 | 4,781.4 / 208.6 | 26.822초 |

입력 수치는 text token이다. D의 물리 prefix 슬롯은 별도 기록하며 정보가 없는 경우 attention을 마스킹한다. 로드·토크나이저 캐시·입력 처리 시간도 JSON에 따로 있다. 서로 다른 GPU와 동시 작업이 있어 정밀한 처리량 벤치마크로 일반화하지 않는다.

품질 비교기는 규칙 fallback을 실행하지 않는다. 표의 필요율은 `fallback_required_rate`이며 실제 사용률 `fallback_used_rate`는 네 조건 모두 0이다. 기존 `fallback_rate`는 필요율의 호환 별칭이다. 재집계본에서 이를 명시했고, 실제 CLI의 보고서는 Qwen 실패 후 규칙을 실행한 경우에만 `execution.fallback_used=true`를 기록한다.

**B/C 입력·렌더링 prompt·구현·디코딩 해시는 7 view 모두 일치**했다. C/D도 base, 16 step, seed, source group, 공통 입력·학습 target이 같음을 확인했다. A/B는 출력 프로토콜도 달라 입력만의 효과를 분리한 실험은 아니다.

공통 화성의 지지 주장이 있어도 음정 진행의 낮은 점수 등 반대 근거가 있으면 `low_similarity`를 허용하도록 검증기를 보완했다. 이 회귀 검사 후 저장된 B/C/D 출력 **23개**의 유효성 판정이 기존 실행과 같음을 `current_validator_audit.json`에 확인했다.

유일 근거 제거에서 A/B는 보류하지 못했고 C/D는 `insufficient_evidence`로 반응했다. 원래 단일 후보의 순서 변경은 자명한 입력이므로 제외했다. 추가 **실제 후보 두 개의 순서 검사**에서 B는 0/2, C는 2/2 응답 검증을 통과했다. C의 판단은 유지됐지만 인용 ID 집합은 달랐다. 두 후보에서 허용되는 주장을 골랐는지와 ID의 단순 일치를 구분한다. 이 또한 한 원곡 그룹의 검사다.

위 지지율은 **구조화된 주장 계약의 자동 검사**다. 모든 자연어 문장의 의미 정확도가 아니다. 사람 판단 정답과 의미 평가가 **0건이므로 사람 판단 accuracy·macro-F1·의미 충실도는 `null`**로 남긴다. 학습이 임시 정책·출력 형식을 습득한 결과를 실제 음악 분별 성능 향상으로 해석하지 않는다.

마스킹 실험의 `abc_v2.json`과 `d_v2.json`도 보존했다. C/D는 각 2/6만 통과했다. 가용성 기반 보류 라벨만 감독하자 지지 주장이 있으면서 `insufficient_evidence`를 고르는 편향이 관찰됐다. 이를 숨기지 않고, 임시 정책 smoke와 기본 마스킹 경로를 분리했다. 사람이 검수한 균형 있는 판단 라벨이 필요하다.

### Soft token과 실제 오디오 개입

기본 prefix는 제거하거나 A/B 특징을 교환해 추가 생성했다. D의 prefix 제거 응답은 원래 응답과 달라졌지만 주장·판단 모순으로 검증에 실패했다. 따라서 유효한 품질 개선 사례로 세지 않았다. A/B 교환의 판단·설명은 같았다. 복사된 패턴 한 사례의 대칭 검사이며 다른 음악 내용을 교환해 이해 능력을 입증한 결과가 아니다. 기본 7 view에서 C보다 나은 품질을 입증하지 못했고 D는 선택 실험으로 유지한다.

`audio_reanalysis/summary.json`은 **WAV 자체를 수정한 별도 결과**다. B 15–25초의 알려진 삽입 부분을 새 파일에서 무음으로 만들고 MERT·SongFormer·LoRA를 다시 실행했다. 두 경우 모두 Qwen은 `executed_validated`/`review`였다. 원래 삽입 위치에 대한 top-1 양쪽 최소 IoU는 **1.0 → 0.0**이었다. 다른 후보가 남을 수 있어 전역 판단의 반전을 정답으로 강제하지 않았다. 원음 파일은 보존했고 원본·수정본 SHA를 기록했다.

### 디코더와 보정

실제 tokenizer의 `im_end=248046`과 모델 기본 EOS `248044`의 불일치를 고쳤다. LMFE 자유 문자열 캐시의 JSON 경계 토큰과 Unicode alphabet 덮어쓰기 문제도 실제 토크나이저 검사로 확인·수정했다.

같은 입력 **6,016 token**, 같은 LoRA, 원본 사례 한 개에서 설정 하나씩 바꿨다. 네 조건 모두 1/1 자동 검증 통과이며 의미 정답은 없다.

| 설정 | 출력 token | 생성 시간 | tokenizer 캐시 준비 |
| --- | --- | --- | --- |
| 기본 | 243 | 28.634초 | 8.846초 |
| JSON 제약 해제 | 243 | 24.154초 | 0초 |
| Unicode 문자 범위 | 243 | 59.234초 | 8.053초 |
| 네 token 반복 제한 | 247 | 29.088초 | 8.326초 |

`decoding/summary.json`에서 입력 동일성을 확인했다. 한 사례·반복 측정 없는 엔지니어링 결과이므로 제약을 없애는 것이 일반적으로 안전하거나 낫다고 결론짓지 않는다.

`calibration_windows.json`은 validation 원곡 **2개**의 정렬 창에 통제 라벨을 적용했다. 요소별 유효 표본은 10/10/8/10이며 방법·양성/음성 수·임계값·checkpoint SHA를 기록했다. 확률 보정이나 분석 정확도 검증은 아니다. 과거 pair-maximum 보정 시도 `calibration.json`은 최종 창 단위 보정과 구분한다.

### 재현 명령·산출물

```bash
.venv-clean/bin/python -m pytest -q --disable-warnings
.venv-qwen/bin/python -m unittest discover -s tests -p test_qwen_decoding.py -v
.venv/bin/python scripts/qwen_dataset.py prepare --manifest data/synthetic/manifest.json --backend ours --checkpoint checkpoints/gpu/mert95_smoke.pt --device cuda --max-per-split 2 --evidence-limit 1 --time-bins 4 --reports results/smoke_20261105/integrated_gpu_report.json --out data/qwen_pilot_v2
.venv-qwen/bin/python scripts/train_qwen.py --dataset data/qwen_pilot_v2/pilot.jsonl --config configs/qwen/lora_smoke.json --max-steps 16 --out checkpoints/qwen/lora_v2 --device cuda
.venv/bin/python scripts/prepare_soft_cache.py --dataset data/qwen_pilot_v2/pilot.jsonl --checkpoint checkpoints/gpu/mert95_smoke.pt --device cuda --out data/qwen_soft_v2
.venv-qwen/bin/python scripts/train_qwen.py --dataset data/qwen_soft_v2/pilot_with_soft.jsonl --config configs/qwen/lora_smoke.json --soft-config configs/qwen/soft_tokens.json --max-steps 16 --out checkpoints/qwen/soft_v2 --device cuda
```

임시 정책 라벨을 쓴 별도 실행은 `lora_weak_policy_smoke.json`과 명시적으로 수정한 `pilot_weak_policy.jsonl`을 사용했다. 새 자료를 준비할 때는 `prepare --supervise-rule-verdict`로 같은 라벨 정책을 선택할 수 있다. 비교는 `evaluate_qwen.py`, 의미 검수 후 재집계는 `score_qwen_outputs.py`, 실제 오디오 수정은 `evaluate_audio_interventions.py`로 실행했다. 상세 명령은 [확장 안내](QWEN_ALIGNMENT.md)에 있다.

기존 CLI의 DSP 보고서와 기존 MERT 체크포인트 추론을 다시 실행했다. `integrated_claims.json`은 전체 WAV + SongFormer + LoRA에서 `review`, 근거 ID와 연결된 주장 네 개·한국어 설명을 저장했다. 기존 `qwen_worker.py`도 어댑터를 새로 로드해 `insufficient_evidence` 응답을 생성했다. 최종 오디오 재분석 보고서는 원래 bin 참조를 포함한 최신 근거 형식으로 생성했다.

자료·가중치·실험 결과·검수 화면은 Git 제외다. 읽기용 실제 검수 파일은 `data/qwen_review/index.md`다. 검수 데이터 본 학습, 사람 의미 평가, 선호 최적화, soft token의 음악 내용 기여 일반화는 실행하지 않았다.

## 11월 5일 재개 후 검증

독립 CPU 환경에서 기본 테스트 **12개 통과**를 확인했다. 별도 Qwen 환경에서는 실제 공개 토크나이저로 **디코더 테스트 5개를 모두 통과**했다. 다중 근거 ID, 한국어 문장, 숫자·이스케이프 차단, 반복 차단, 캐시/문자 검사 불일치와 불완전 UTF-8 우회에 대한 회귀 검사다. 모델 가중치는 이 토크나이저 검사에 사용하지 않는다. 디코더 의존성이 없는 기본 환경에서는 이 다섯 테스트를 건너뛴다.

Qwen 입력에서 긴 예측 분포를 최빈 클래스와 상위 크로마 등으로 요약하고 원본 근거는 보고서에 보존한다. 출력의 한국어 비율·반복을 검사하며, 설명에만 네 토큰 반복 제한을 적용한다. LMFE의 자유 문자열 캐시와 문자 parser가 서로 다른 문자를 허용해 JSON을 중간에 끊는 문제를 수정했다. 광범위한 정규식을 매 토큰마다 탐색하는 대신 캐시를 재사용한다. 이 수정 전의 반복 문장과 잘린 JSON은 성공으로 집계하지 않았다.

```bash
.venv-clean/bin/python -m pytest -q --disable-warnings
.venv-qwen/bin/python -m unittest discover -s tests -p test_qwen_decoding.py -v
.venv-clean/bin/python -m compileall -q src scripts
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m src compare \
  data/synthetic/insertion/test/song_a.wav data/synthetic/insertion/test/song_b.wav \
  --backend ours --checkpoint checkpoints/gpu/mert95_smoke.pt \
  --songformer-python .venv-songformer/bin/python --judge qwen \
  --qwen-python .venv-qwen/bin/python --device cuda \
  --out results/smoke_20261105/integrated_gpu_report.json
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m src evaluate \
  --manifest data/synthetic/manifest.json --backend dsp --judge qwen \
  --qwen-python .venv-qwen/bin/python --device cuda --judge-cases 1 \
  --out results/smoke_20261105/qwen_gpu_eval.json
```

GPU 비교에서 생성한 MERT/SongFormer 원본 근거를 보존하고, 디코더 수정 후에는 `judge_pair`로 Qwen 단계만 재실행했다. 위 비교 명령은 전체 흐름을 재현하는 명령이다. GPU 번호는 이 환경의 실행 예시이며 사용 가능한 장치를 선택한다.

최종 통합 보고서의 Qwen은 **`executed_validated`**, `cuda:0`/`torch.bfloat16`, 고정 **Qwen3.5-0.8B**, non-thinking으로 실행됐다. 첫 응답에서 스키마와 ID 검증을 통과했다. 모델 로드 **8.443초**, 토크나이저 캐시 **9.590초**, 생성 **99.415초**, Qwen 단계 전체 **117.448초**였다. 이 시간은 MERT/SongFormer 실행을 제외한다. 선택적 `causal_conv1d`/`flash-linear-attention` 커널 없이 참조 구현으로 실행했다.

MERT 체크포인트 재로딩과 SongFormer의 양쪽 전체 곡 분석이 성공했고, 최상위 대응은 **A 10–20초 / B 15–25초**였다. Qwen은 `high_similarity`와 실제 근거 ID 세 개를 반환했다. **수동 검토에서는 같은 설명 안에서 높은 유사성을 주장하다가 그 판단 기준이 충족되지 않는다고 서술하는 모순을 확인했다.** 따라서 이 응답은 JSON/ID 처리 성공 사례이며, 음악적 설명의 충실도 성공 사례로 간주하지 않는다. 사용자에게 보이는 정확한 시간·점수는 원본 근거에서 가져온다. 이 모델의 설명 품질과 판단 보정은 남은 연구 과제다.

최종 `qwen_gpu_eval.json`은 **삽입 사례 한 개의 원본/교환 응답 두 개**를 평가했다. 스키마·알려진 ID 검증 **2/2**, 첫 응답 검증 **2/2**, 판단 일관성 **1/1**, 인용 ID 집합 일관성 **1/1**이었다. 양쪽 모두 `high_similarity`, `E001/E002/E003`을 반환했고 규칙 fallback을 사용하지 않았다. Qwen 단계 전체 시간은 원본 **98.356초**, 교환 **103.184초**였다. 이 평가는 DSP에서 얻은 동일 근거의 A/B 시간·음악 정보·구조 맥락을 바꾸는 검사다. 표본 한 개의 형식·순서 일관성 결과를 음악적 판단 정확도나 설명 충실도 성능으로 일반화하지 않는다.

## 11월 4일 중지 당시 기록

사용자 요청으로 실행 중인 검증 프로세스를 종료하고 `502b4f0`으로 커밋했다. 당시 CPU 기준 테스트 9개와 오프라인 MVP 실행은 통과했다. 권한 확장 실행에서는 RTX A6000의 CUDA 연산, MERT 95M 헤드 학습 2 epoch 및 마지막 두 블록 미세조정 1 epoch까지 확인했다. GPU의 선형 보간 역전파에는 비결정적 연산 경고가 있어 비트 단위 재현을 보장하지 않는다.

JSON 디코더의 정규식 최적화는 코드에 반영했지만, 해당 최종 버전의 Qwen A/B 검사와 전체 GPU 통합 보고서 생성은 중지 시점에 완료되지 않았다. 이를 성공 결과로 간주하지 않는다. `reference_files/`는 로컬에 보존하며 Git 추적에서 제외한다.

## 11월 4일 기준 범위와 결과

- 시스템 패키지를 공유하지 않는 `.venv-clean`에서 설치, `python -m pytest -q --disable-warnings`, `PYTHON=.venv-clean/bin/python bash scripts/run_smoke.sh`를 실행했다. **9개 테스트 통과**, 합성 생성·작은 학습·체크포인트 저장/재로딩·WAV 추론·JSON 평가가 성공했다.
- 공개 MERT 95M에서 1초 음원의 `(1,20,768)` 프레임 특징이 유한한 값을 반환했다. MERIT는 공식 330M과 공개 헤드의 실제 조합으로 실행했다.
- 고정 MERT 헤드 학습, 예측 손실 없는 ablation, 마지막 두 블록 미세조정을 각각 실행했다. 체크포인트 로드 후 음원 비교도 실행했다.
- 공식 SongFormer가 합성 곡 A 전체를 verse 구간으로, B를 verse/inst/verse 구간으로 반환했다. 원본 종료 시간과의 작은 프레임 격자 차이는 범위 검사 후 잘랐다.
- 로컬 Qwen은 한국어 JSON 생성과 ID 검증을 실행했다. 실패한 시도·형식 재요청·규칙 fallback은 보고서에 구별한다. 최종 디코더의 A/B 재검사는 완료 전에 중지했다.

## 가져온 부분과 새 구현

MERIT의 공개 projection head와 공식 층 평균/연결 방식, MERT·SongFormer·Qwen의 공식 로딩 API를 적용했다. 원본 커밋·파일·변경 내용과 라이선스 전문은 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)에 기록했다.

전체 음원 시간·패딩 관리, 네 시간축 헤드와 두 종류 손실, 정답 마스크, 원곡 분할·통제 렌더링·변형, POP909/Slakh 로더, 전체 창 비교·후보 묶기, 근거 검증·규칙 기준선·평가 CLI는 새로 작성했다. 실제 자료에서 발견한 Slakh action-map 형식과 saved/rendered 플래그 불일치도 처리했다.

## 직접 실행한 주요 명령

```bash
PYTHON=.venv-clean/bin/python bash scripts/run_smoke.sh
.venv/bin/python scripts/prepare_models.py mert95 mert330 merit_heads songformer qwen
.venv/bin/python -m src train --manifest data/synthetic/manifest.json --backbone mert95 --max-pairs 8 --epochs 2 --batch-size 2 --smoke --out checkpoints/mert95_smoke.pt
.venv/bin/python -m src train --manifest data/synthetic/manifest.json --backbone mert95 --max-pairs 8 --epochs 2 --batch-size 2 --prediction-weight 0 --smoke --out checkpoints/mert95_ablation_smoke.pt
.venv/bin/python -m src train --manifest data/synthetic/manifest.json --backbone mert95 --max-pairs 2 --epochs 1 --batch-size 1 --unfreeze-last 2 --init-checkpoint checkpoints/mert95_smoke.pt --smoke --out checkpoints/mert95_finetune_smoke.pt
.venv/bin/python -m src compare data/synthetic/insertion/test/song_a.wav data/synthetic/insertion/test/song_b.wav --backend ours --checkpoint checkpoints/mert95_smoke.pt --songformer-python .venv-songformer/bin/python --judge qwen --qwen-python .venv-qwen/bin/python --out results/smoke_20261104/integrated_report.json
.venv/bin/python -m src evaluate --manifest data/synthetic/manifest.json --backend merit --out results/smoke_20261104/merit_eval.json
.venv/bin/python -m src evaluate --manifest data/synthetic/manifest.json --backend ours --checkpoint checkpoints/mert95_smoke.pt --out results/smoke_20261104/ours_mert_eval.json
.venv/bin/python -m src evaluate --manifest data/synthetic/manifest.json --backend ours --checkpoint checkpoints/mert95_ablation_smoke.pt --out results/smoke_20261104/ablation_eval.json
.venv/bin/python -m src evaluate --manifest data/synthetic/manifest.json --backend dsp --judge qwen --qwen-python .venv-qwen/bin/python --judge-cases 1 --out results/smoke_20261104/qwen_eval.json
```

## 소규모 학습 확인

| 실행 | train/validation 창 쌍 | 마지막 epoch train/validation 손실 | 파라미터 변경 |
| --- | --- | --- | --- |
| mert95_smoke | 8/8 | 1.3061/1.3952 | True |
| mert95_ablation_smoke | 8/8 | 0.0235/0.0399 | True |
| mert95_finetune_smoke | 2/2 | 1.0608/1.0789 | True |
| mert95_cache_smoke | 2/2 | 1.7874/1.6558 | True |
| pop909_small | 6/6 | 1.6833/1.5778 | True |
| pop909_slakh_small | 12/6 | 1.5215/1.2740 | True |
| GPU mert95_smoke | 8/8 | 1.3061/1.3952 | True |
| GPU finetune_smoke | 2/2 | 1.0526/1.1195 | True |

손실에는 서로 다른 항이 들어가므로 예측 손실 포함 모델과 ablation의 전체 손실 크기를 직접 비교하지 않는다. validation으로 선택한 체크포인트의 epoch는 마지막 epoch와 다를 수 있다. 미세조정 전후의 실제 백본 텐서를 비교해 **10·11번 블록의 32개 텐서만 변경**되고 다른 블록은 동일함을 확인했다. `finetune_parameter_audit.json`에 기록했다.

## 합성 test 평가

test 원곡 2개에서 파생된 쌍 **10개**, 화성은 불확실한 조옮김 2개를 제외한 **8개**다. 아래 수치는 작은 통제 데이터의 실행 확인이며 실제 음악 성능 추정이 아니다. 임계값은 보정하지 않은 **0.8**이다.

| 백엔드 | 음정/선율 AUC | 리듬 AUC | 화성 AUC | 반주·베이스 AUC | 어려운 음성 쌍 음정/선율 FPR |
| --- | --- | --- | --- | --- | --- |
| DSP 프록시 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MERIT 330M | 0.917 | 0.250 | — | — | 0.000 |
| MERT 95M + 음악 정보 예측 | 0.875 | 1.000 | 1.000 | 0.875 | 1.000 |
| MERT 95M ablation | 0.917 | 0.750 | 1.000 | 0.500 | 1.000 |

MERIT의 선율과 우리의 음정 진행은 동일 과제가 아니므로 위 선율 수치의 직접 우열을 주장하지 않는다. 엄밀한 공통 요소는 리듬이며, 음색은 반주·베이스와 비교하지 않았다. 작은 학습 모델의 기본 임계값 오탐률이 높아 실제 사용에 앞서 validation 보정과 더 많은 학습·평가가 필요하다.

| 백엔드 | 삽입 top-1 양쪽 최소 IoU | 후보 중 최대 양쪽 최소 IoU |
| --- | --- | --- |
| dsp_eval | 1.000 | 1.000 |
| merit_eval | 1.000 | 1.000 |
| ours_mert_eval | 1.000 | 1.000 |
| ablation_eval | 1.000 | 1.000 |

삽입 test는 **1사례**다. 정답 A 10–20초, B 15–25초와 동일한 WAV 패턴을 복사해 넣은 조건이다. 긴 연결 후보가 정확한 개별 창을 덮지 않도록 peak window와 monotone chain을 함께 보존한다. 실제 곡의 경계 정밀도나 변형 삽입의 재현율이 검증됐다는 뜻은 아니다.

## 실제 공개 자료 확인

- `fetch_reference_examples.py`: Kevin MacLeod의 **Vibe Ace** 원본과 CC-BY-3.0 크레딧, POP909 MIDI·주석 3곡, 공식 Slakh redux 파일을 저장했다. 원본 녹음의 겹치는 24초 발췌 두 개를 MERIT로 비교해 `merit_recording_report.json`을 생성했다. 같은 녹음에서 잘라낸 예제로 독립 곡 평가와 구별한다.
- POP909는 원곡 ID 분할 후 첫 구간을 렌더링해 실제 95M 백본으로 1 epoch 학습했다. 첫 구간의 주선율 정답이 비어 있으면 해당 손실과 정보 출력을 마스킹한다.
- BabySlakh 공식 v2 스트림에서 Track00020 한 곡, 파일 23개를 추출했다(전체 압축 다운로드/검증과 구별). 실제 **베이스 음표 336개**, **악기 스템 10개**, 베이스 WAV를 24 kHz 모노로 디코딩한 약 **347.754초** 길이를 확인했다. metadata flag와 실파일의 불일치 20건을 기록했다.
- 공식 redux action-map 2,100개 항목 중 omit을 제외한 1,710개를 확인했고, 해당 파일로 샘플의 train 소속을 적용했다. 샘플 한 곡으로 새 validation/test를 만들지 않았다.
- POP909 + 실제 Slakh 베이스 샘플을 함께 준비한 작은 manifest에서도 학습을 실행했다. 원래 mix/stem 파형으로 학습한 실험은 아니며, 통제 MIDI 재렌더링 학습이다.

## 숫자와 설명의 출처

실행 과정에서 소형 Qwen이 구간 시간을 합쳐 쓰거나 입력 점수를 과장하는 오류를 확인했다. 최종 구현은 설명에서 근거 ID 외 숫자를 허용하지 않고, 시간·요소별 수치는 원본 evidence에서 표시한다. 원본 근거의 보존과 문장 내용의 충실도 검증을 구분한다.

## 현재 구현에서 남은 검증

- 전체 코퍼스 본 학습은 실행하지 않았다. `scripts/train_full.sh`에 재현 명령을 제공했다.
- MERT의 요소 분리 성능, 실제 상용 음악 혼합 음원에서의 주선율·베이스 추정과 유사성 성능은 추가 검증이 필요하다.
- SongFormer 구간 라벨은 실행을 확인했으며 정답 정확도와 세밀한 프레이즈 정렬은 평가하지 않았다. 프레이즈 유사도 수치는 unavailable로 남긴다.
- Qwen의 스키마·ID·순서 교환 검사는 자연어 설명의 모든 주장을 보증하지 않는다. 작은 모델이 점수·구조 상태를 잘못 서술할 수 있어 설명 충실도와 사람 평가가 필요하다.
- 모든 모델·데이터·실험 파일은 Git에서 제외하고 이 검증 요약과 코드·설정·출처만 추적 대상으로 남겼다.

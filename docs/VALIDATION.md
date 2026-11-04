# 실행 검증 기록

확인일: **2026-11-04, Asia/Seoul**. 기본 샌드박스에서는 GPU 장치가 보이지 않아 먼저 CPU 기준 검증을 수행했다. 이후 권한 확장 실행에서 RTX A6000 48GB GPU를 확인해 추가 소규모 검증을 수행했다. MERIT 330M, 우리 백본 MERT 95M, SongFormer, Qwen3.5-0.8B의 크기를 유지했다. 환경 버전은 [environment_versions.json](environment_versions.json)에 기록했다.

## 작업 중지 시점

사용자 요청으로 실행 중인 검증 프로세스를 종료하고 현재 상태를 커밋한다. CPU 기준 테스트 9개와 오프라인 MVP 실행은 통과했다. 권한 확장 실행에서는 RTX A6000의 CUDA 연산, MERT 95M 헤드 학습 2 epoch 및 마지막 두 블록 미세조정 1 epoch까지 확인했다. GPU의 선형 보간 역전파에는 비결정적 연산 경고가 있어 비트 단위 재현을 보장하지 않는다.

JSON 디코더의 정규식 최적화는 코드에 반영했지만, 해당 최종 버전의 Qwen A/B 검사와 전체 GPU 통합 보고서 생성은 중지 시점에 완료되지 않았다. 이를 성공 결과로 간주하지 않는다. `reference_files/`는 로컬에 보존하며 Git 추적에서 제외한다.

## 범위와 결과

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

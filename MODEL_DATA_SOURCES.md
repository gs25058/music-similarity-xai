# 모델·데이터 버전과 이용 조건

확인일 2026-11-05. 코드 출처·수정 고지는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 따로 기록했다. 모든 외부 가중치는 고정 리비전을 사용한다.

Qwen 확장에서는 같은 `Qwen3.5-0.8B` 리비전에 선택적 LoRA를 학습한다. 원본 가중치는 고정하고 어댑터·연결층만 별도 저장한다. 자동 파일럿의 예측·주장·판단·설명 JSONL은 원래 manifest와 source group, 원본 보고서 SHA, 검수 상태, 라벨 출처를 보존한다. `rule_provisional`과 근거 가용성의 정의 기반 라벨은 사람의 음악적 판단 정답이 아니다. 실제 학습 데이터 SHA·사용 분할·검수 필터·설정은 각 어댑터의 `training_meta.json`에 기록한다. 가중치·데이터·캐시는 계속 Git에서 제외한다.

| 모델 | 고정 리비전 | 공식 출처의 라이선스 표시 | 용도 |
| --- | --- | --- | --- |
| MERT-v1-95M | `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5` | [CC-BY-NC-4.0](https://huggingface.co/m-a-p/MERT-v1-95M) | 우리 모델의 공유 인코더. 768차원 프레임 특징, 고정 후 마지막 두 블록 해제 가능. |
| MERT-v1-330M | `5240c2708a5acaee1007f43fb9735c7dcd0b78c9` | [CC-BY-NC-4.0](https://huggingface.co/m-a-p/MERT-v1-330M) | MERIT 공식 백본, 1024차원 5개 층 연결. |
| MERIT 공개 헤드 | `a85df30eca1ba112eb594285f3ba1d96488e7883` | [MIT](https://huggingface.co/amaai-lab/merit) | 선율·리듬·음색 128차원 헤드. 화성·베이스·구조 헤드로 해석하지 않음. |
| SongFormer | `a75880ed1b7375ac71860ec6c4fc9c899cf99515` | [HF 카드](https://huggingface.co/ASLP-lab/SongFormer)에 별도 라이선스 태그 미표시. 공식 GitHub 코드의 CC-BY-4.0과 구분. | 공식 전체 곡 추론 패키지. 기능 라벨·경계는 비교 점수와 분리된 맥락. 내부 MuQ/MusicFM 관련 출처는 다운로드 패키지의 고지를 보존. |
| Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | [Apache-2.0](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 로컬 한국어 판단. 가중치 고정, non-thinking, 비양자화. |
| SongFormer의 Conformer 설정 | `6b36ef01c6443c67ae7ed0822876d091ab50e4aa` | [Apache-2.0](https://huggingface.co/facebook/wav2vec2-conformer-rope-large-960h-ft) | MusicFM 생성 시 읽는 config.json만 사용. 추가 사전학습 가중치는 사용하지 않음. |

`scripts/prepare_models.py`는 `models/<component>/artifact_version.json`도 기록한다. 사전학습 저장소 전체나 MERT fairseq 체크포인트는 필요하지 않다. HF의 동적 Python 코드는 명시한 고정 리비전의 파일만 내려받아 로컬에서 실행한다.

| 데이터 | 버전·출처 | 이용 조건 기록 | 로더·변형 정책 |
| --- | --- | --- | --- |
| 합성 데이터 | `src/synthetic.py`, seed와 원곡 ID 분할을 manifest에 기록 | 이 프로젝트가 생성한 점수와 oscillator 음원 | 요소별 통제 쌍·어려운 음성 쌍·양쪽 삽입 시간 정답. 실제 음악 성능의 대체 평가가 아님. |
| POP909 | [공식 저장소](https://github.com/music-x-lab/POP909-Dataset/tree/d83e6edba6872a704f5d3b8b32f5cb540088dae6), `d83e6edba6872a704f5d3b8b32f5cb540088dae6` | 저장소 LICENSE는 MIT(Copyright 2020 Music X Lab). 원곡 녹음·작곡의 권리와 별개이며 원곡 음원은 포함하지 않음. | MIDI와 알고리즘 주석을 읽음. 피아노 저음 라벨은 파생 근거, 실제 베이스 연주 정답과 구분. |
| Slakh2100 | [공식 사이트](https://www.slakh.com/), slakh-utils `3f62d6a4b0e5952237dd5178ac3513e42161ec0a`의 `splits/redux.json` | [공식 배포 안내](https://github.com/ethman/Slakh#license-and-attribution)의 **CC-BY-4.0**. slakh-utils 코드의 MIT와 구별. | 공식 redux action-map의 omit/move를 적용하고 UUID 검사. 베이스 MIDI·스템 음원 경로·원본 mix 경로 반환. 전체 데이터 자동 다운로드 없음. |
| BabySlakh | [공식 v2, DOI 10.5281/zenodo.4603870](https://zenodo.org/records/4603870), 16 kHz | Slakh의 CC-BY-4.0, Ethan Manilow, Gordon Wichern, Prem Seetharaman, Jonathan Le Roux 저작자 표시 | `fetch_slakh_sample.py`는 압축 스트림의 첫 곡만 추출한다. 선택 파일별 SHA-256을 저장하며 전체 압축 파일의 해시 검증과 구별. |
| 공개 녹음 실행 예제 | librosa의 [공식 예제 목록](https://librosa.org/doc/0.11.0/generated/librosa.util.example.html): Kevin MacLeod, *Vibe Ace* | 다운로드한 `Vibe_Ace.txt`의 출처·라이선스를 원문 그대로 보존 | 같은 녹음의 겹치는 두 발췌로 디코딩·MERIT 실행만 확인. 독립된 곡의 유사성 평가 결과로 사용하지 않음. |

원곡·변형 WAV/MIDI, 체크포인트, HF 캐시, 실험 결과는 `.gitignore`로 제외한다. 학습 manifest에는 원곡 ID, 분할, 변형 종류, 요소별 `null` 마스크, 정답 출처를 기록한다. 재현 시 해당 manifest와 데이터·라이선스 파일을 함께 보관한다.

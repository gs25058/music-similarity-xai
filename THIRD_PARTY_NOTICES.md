# 외부 코드 출처 및 변경 기록

확인일: 2026-11-05. 코드 라이선스와 모델·데이터 이용 조건은 구분한다. 모델 리비전·데이터 출처는 [MODEL_DATA_SOURCES.md](MODEL_DATA_SOURCES.md), 실제 가중치 다운로드 명세는 [configs/model_versions.json](configs/model_versions.json)에 있다.

## Qwen 학습·검증 확장

- **PEFT 0.21.2**: [공식 v0.21.2](https://github.com/huggingface/peft/tree/86b6cbda4c125337bdc0cc7fc3f2fe09c17238e1), 커밋 `86b6cbda4c125337bdc0cc7fc3f2fe09c17238e1`. `peft/mapping_func.py`, `peft/peft_model.py`, `peft/tuners/lora/config.py`의 공개 `get_peft_model`, `LoraConfig`, `PeftModel.from_pretrained`, 저장 API를 패키지 의존성으로 사용한다. `scripts/qwen_runtime.py`, `src/qwen_training.py`, 기존 `scripts/qwen_worker.py`에 적용했다. 라이브러리 소스 복사·수정은 없으며 [Apache-2.0 원문](third_party/licenses/PEFT-Apache-2.0.txt)을 보존했다. Accelerate 1.15.0은 로딩 의존성으로 설치했으며 분산 학습 코드를 가져오지 않았다.
- **LMFE 어댑터 추가**: 아래에 기록한 동일 0.11.3 API를 사용한다. 자유 문자열 캐시에서 빠지던 `.",` 등 구두점·닫는 따옴표·JSON 구분자를 포함한 토큰을 제한된 후보 집합에서 실제 parser로 검사하는 `ClosingBoundaryTokens`를 새로 구현했다. Unicode 조건에서는 `TokenEnforcer`가 바꾼 parser 문자 범위를 복원한다. 설치한 패키지는 수정하지 않았다.
- **Qwen 공식 API 추가 사용**: 동일 고정 리비전의 모델과 tokenizer를 그대로 사용한다. tokenizer의 assistant-turn 종료 ID와 모델 기본 종료 ID가 달라 두 worker의 `generate`에 실제 tokenizer 종료 ID를 명시했다. 음악 모델과 vision 분기를 고정하고 확인한 text attention 모듈에만 LoRA를 적용한다.
- 시간순 근거, 양쪽 NMS, 보정 단위 검사, 주장 계약·검증, source-group JSONL·검수 도구, 출력 토큰 마스킹·가중 손실, feature cache·projection·prefix, 충실도 평가·오디오 재분석 루프는 이 프로젝트의 새 구현이다.

설계 참고는 [MusiLingo 논문](https://aclanthology.org/2024.findings-naacl.231/), [SelfCite 논문](https://proceedings.mlr.press/v267/chuang25a.html)과 [공식 저장소](https://github.com/facebookresearch/SelfCite), [LoRA 논문](https://arxiv.org/abs/2106.09685)이다. MusiLingo의 모델 연결 관점과 SelfCite의 맥락 제거·유지 평가 관점을 참고했으며 해당 연구의 학습 코드는 복사하지 않았다. 자체 reward·선호 최적화를 실행하거나 논문의 성능을 재현했다고 주장하지 않는다.

## 가져오거나 수정한 코드

| 출처와 고정 커밋/리비전 | 원본 파일 | 반영 파일 | 가져온 부분과 변경 |
| --- | --- | --- | --- |
| [AMAAI-Lab/MERIT](https://github.com/AMAAI-Lab/MERIT/tree/aa0039b4ead7d451b5af9a9f3b4b7c206feaed51), `aa0039b4ead7d451b5af9a9f3b4b7c206feaed51` | `README.md` Quick Inference, `evaluation/encode_folder.py`의 인코딩·구간 처리 | `src/backends.py`, `src/audio.py` | 공개 ProjectionHead 구조, 체크포인트 필드, 330M의 `(3,4,5,6,23)` 층 평균·연결 방식을 재사용/수정. torchaudio 대신 soundfile/scipy 사용. 30초 절단·비중첩 창을 전체 곡 10초/5초 창으로 변경. 마지막 창은 실제 길이만 인코딩하고 원곡 시간·패딩 길이를 별도 보존. 미지원 요소에 다른 헤드를 대응시키지 않음. |
| [MERT 모델 카드](https://huggingface.co/m-a-p/MERT-v1-95M/tree/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5), 95M `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`; 330M `5240c2708a5acaee1007f43fb9735c7dcd0b78c9` | 모델 카드 `README.md` 로딩 예제 | `src/backends.py` | AutoModel와 Wav2Vec2FeatureExtractor 사용법 적용. 프레임 특징 래퍼, 시간 해상도 정렬, 마지막 두 블록 해제, 정확한 가중치 로드 검사 추가. PyTorch weight_norm의 직렬화 키 두 개만 변환하며 텐서·모델 크기를 바꾸지 않음. 다운로드된 `modeling_MERT.py`, `configuration_MERT.py`는 `models/`에 보존하며 Git에는 포함하지 않음. |
| [SongFormer 공식 모델 카드](https://huggingface.co/ASLP-lab/SongFormer/tree/a75880ed1b7375ac71860ec6c4fc9c899cf99515), `a75880ed1b7375ac71860ec6c4fc9c899cf99515`; [공식 코드](https://github.com/ASLP-lab/SongFormer/tree/139b2aa3b14bd1c6d961d0994e9fc975f1ef7fd5), `139b2aa3b14bd1c6d961d0994e9fc975f1ef7fd5` | `README.md` 추론 예제 | `scripts/songformer_worker.py` | 공식 경로 설정, AutoModel 로딩, 전체 음원 forward를 재사용. 별도 Python 프로세스의 JSON 통신, CPU/GPU 인자, stdout 분리, 시간 스키마 검증 추가. 모델 소스는 수정하지 않고 `models/songformer/`에서 실행. |
| [Qwen 공식 모델 카드](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17), `2fc06364715b967f1860aea9cf38778875588b17` | 모델 페이지의 Transformers 로딩·chat template 예제 | `scripts/qwen_worker.py` | 공식 AutoProcessor/모델 로딩과 chat template 사용. 로컬 파일, non-thinking, 고정 가중치, 근거 JSON 입력, 스키마 검증 후 최대 1회 형식 재요청 추가. 모델 구조·크기·정밀도 축소 없음(CPU float32, GPU bfloat16). |

MERIT 코드는 **MIT**, Copyright (c) 2026 AMAAI Lab이다. 원문 저작권·허가·면책 전문을 [third_party/licenses/MERIT-MIT.txt](third_party/licenses/MERIT-MIT.txt)에 포함했다.

MERT 공식 코드 저장소의 라이선스는 Apache-2.0이며 [전문](third_party/licenses/MERT-code-Apache-2.0.txt)을 보존했다. 모델 카드의 가중치 라이선스 CC-BY-NC-4.0은 별도로 적용된다.

SongFormer 공식 GitHub 코드의 표시 라이선스는 CC-BY-4.0이며 [원문 고지](third_party/licenses/SongFormer-code-CC-BY-4.0.txt)를 보존했다. 저자: Chunbo Hao, Ruibin Yuan, Jixun Yao, Qixin Deng, Xinyi Bai, Yanbo Wang, Wei Xue, Lei Xie. 위 래퍼 변경은 이 프로젝트가 작성했으며 원저자의 보증을 뜻하지 않는다. HF 모델 카드에는 별도 가중치 라이선스 태그가 없어 코드 고지로 가중치 이용 조건을 대체하지 않았다.

Qwen 모델 카드의 표시 라이선스는 Apache-2.0이다. 준비 스크립트가 내려받는 원본 LICENSE는 `models/qwen/LICENSE`에 저장한다.

## 설계·스키마 참고, 코드 복사 없음

Qwen JSON 제약 디코딩은 [noamgat/lm-format-enforcer v0.11.3](https://github.com/noamgat/lm-format-enforcer/tree/05c7dc3a46305e39c1a67c8622c433d8deda48a2) 패키지 API를 사용한다. 고정 커밋은 `05c7dc3a46305e39c1a67c8622c433d8deda48a2`, 참고 파일은 `README.md` Basic Tutorial, `lmformatenforcer/integrations/transformers.py`, `lmformatenforcer/tokenizerprefixtree.py`다. `scripts/qwen_decoding.py`에서 공식 토크나이저 준비·JSON parser API를 사용하고, 새 어댑터가 자유 문자열 캐시와 parser에 같은 한글·영문·구두점 범위를 적용하면서 숫자와 이스케이프 토큰을 제외한다. 불완전 UTF-8 토큰을 조합해 문자 제한을 우회하지 못하도록 설명 캐시에서 해당 토큰도 제외한다. 명시적 enum으로 처리하는 근거 ID의 숫자는 유지한다. 정규식으로 한글 자유 문자열 전체를 탐색하지 않고 기존 캐시를 활용하며, 토크나이저 준비 결과는 재시도에도 재사용한다. 설명 문자열에만 Transformers의 공개 `NoRepeatNGramLogitsProcessor`를 적용하는 어댑터를 새로 작성했다. Transformers v5에서 이동한 `PreTrainedTokenizerBase` 공개 클래스에 런타임 호환 별칭을 제공한다. 설치 패키지 파일이나 모델 가중치는 수정하지 않는다. MIT 라이선스 원문은 [third_party/licenses/lm-format-enforcer-MIT.txt](third_party/licenses/lm-format-enforcer-MIT.txt)에 보존했다. 형식 제약은 설명 내용의 정확도를 보증하지 않는다.

SongFormer worker는 MusicFM의 `Wav2Vec2ConformerConfig.from_pretrained`가 부동 Hub ID를 읽는 부분을 `configs/model_versions.json`의 고정 로컬 config로 연결한다. 원본 `musicfm/model/musicfm_25hz.py` 파일과 모델 텐서는 수정하지 않는다. 설정 출처는 `facebook/wav2vec2-conformer-rope-large-960h-ft`, 리비전 `6b36ef01c6443c67ae7ed0822876d091ab50e4aa`, Apache-2.0이며 추가 가중치를 가져오지 않는다.

| 출처 | 고정 커밋 | 참고 파일·내용 | 우리 구현 |
| --- | --- | --- | --- |
| MERIT | `aa0039b4ead7d451b5af9a9f3b4b7c206feaed51` | `training/train_head.py`, `evaluation/probe_eval.py`: 요소별 학습·probe 평가 분리 | `src/train.py`, `src/evaluate.py`의 손실·마스킹·학습·평가 루프는 새 구현. MERIT의 Circle Loss·학습 코드를 복사하지 않음. |
| [yizhilll/MERT](https://github.com/yizhilll/MERT/tree/391062e4c384aaad4e5a992be339ef70769dbd6f) | `391062e4c384aaad4e5a992be339ef70769dbd6f` | `README.md`, 공식 사전학습 구조 확인 | 사전학습 코드는 가져오지 않음. 공개 모델의 프레임 특징과 고정/부분 미세조정 API 사용. |
| [POP909-Dataset](https://github.com/music-x-lab/POP909-Dataset/tree/d83e6edba6872a704f5d3b8b32f5cb540088dae6) | `d83e6edba6872a704f5d3b8b32f5cb540088dae6` | `README.md`의 MELODY/PIANO/BRIDGE 및 주석 스키마 | 새 로더. BRIDGE 트랙을 verse/chorus 구조 라벨로 전용하지 않음. 저장소 MIT 고지 [전문](third_party/licenses/POP909-repository-MIT.txt). |
| [slakh-utils](https://github.com/ethman/slakh-utils/tree/3f62d6a4b0e5952237dd5178ac3513e42161ec0a) | `3f62d6a4b0e5952237dd5178ac3513e42161ec0a` | `README.md`의 `metadata.yaml`, `splits/redux.json`, UUID 중복 처리 설명 | 새 로더. GM program 32–39와 is_drum으로 실제 베이스 MIDI 선택. redux 분할 필수, UUID 중복 검사. MIT [전문](third_party/licenses/slakh-utils-MIT.txt). |
| [pretty-midi](https://github.com/craffel/pretty-midi/tree/286326f165c19e694533263db38033c71c1c1a2b) | `286326f165c19e694533263db38033c71c1c1a2b` | 공개 패키지 API 확인 | `pretty_midi==0.2.10`을 의존성으로 사용. 소스 복사 없음. MIT [전문](third_party/licenses/pretty-midi-MIT.txt). |

기존 `reference_files/`의 논문은 별도 문헌 자료이며 각 원문의 권리를 따른다. 이 프로젝트의 LICENSE가 외부 논문·가중치·데이터에 재적용되는 것은 아니다.

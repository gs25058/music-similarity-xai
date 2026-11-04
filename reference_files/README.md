# 음악 유사성·설명 가능성 연구 참고자료

사용자가 선정한 선행연구 8편을 원문, 한국어 요약, 연구 적용 메모로 정리했다. 확인일: **2026-11-02 (Asia/Seoul)**.

## 자료 구성

- `papers/`: 원문 PDF. 파일 번호는 아래 표와 일치한다.
- `notes/`: 논문별 서지정보, 핵심 내용, 우리 연구의 활용 방향과 한계.
- [references.bib](references.bib): 8편의 BibTeX 서지정보. 인용 키는 아래 표에 표시했다.
- [sources.json](sources.json): 원문 URL, 저장한 판본, 다운로드 상태, 파일 크기와 SHA-256.
- [research_mapping.md](research_mapping.md): 시스템 구성 요소별 연결과 평가 설계 메모.

논문에서 확인한 내용과 **우리 연구의 제안**을 구분했다. 요약은 공식 초록·서지 페이지와 명시한 원문 구간을 바탕으로 작성했으며, 모든 실험을 재현하거나 구현·모델 가중치를 검증한 결과는 아니다. 원문의 저작권과 이용 조건은 각 출처를 따른다.

## 선행연구 목록

| 번호 | 논문·발표 정보 | 주요 내용 | 우리 연구에 활용할 부분 | 자료 / 인용 키 |
| --- | --- | --- | --- | --- |
| 1 | Hwang & Hwang (2024), *Understanding Human Perception of Music Plagiarism Through a Computational Approach*. ISMIR Late-Breaking Demo 확장 초록; arXiv 공개 2026. | 선율·리듬·코드 진행을 중심으로 인간의 표절 인식을 조사하고 LLM 판단 프레임워크를 제안. | 연구 목적과 시스템 구성이 가장 가까움. 완성된 모델의 성능 검증과 연구 제안을 구별. | [요약](notes/01_hwang_music_plagiarism.md) · [PDF](papers/01_hwang_2024_music_plagiarism_arxiv_v1.pdf) · `hwang2024plagiarism` |
| 2 | Batlle-Roca et al. (2026), *On the Human and Computer Alignment of Attribute-Based Music Matches*. arXiv. | 음악 속성별 인간 판단과 계산 지표의 일치 정도를 분석하고 MATCHA 제시. | 요소별 평가와 인간 평가 설계의 근거. 논문의 다섯 속성은 우리의 다섯 요소와 다름. | [요약](notes/02_batlle_roca_matcha.md) · [PDF](papers/02_batlle_roca_2026_matcha_arxiv_v1.pdf) · `batlleroca2026matches` |
| 3 | Li et al. (2024), *MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training*. ICLR 2024. | 음향·음고 정보를 함께 학습하는 음악 자기지도 표현 모델. | 주 음악 인식 모델의 기반. 요소별 유사성 학습은 우리 연구의 확장 과제. | [요약](notes/03_li_mert.md) · [PDF](papers/03_li_2024_mert_iclr.pdf) · `li2024mert` |
| 4 | Hao et al. (2025; 2026 개정), *SongFormer: Scaling Music Structure Analysis with Heterogeneous Supervision*. arXiv. | 여러 범위의 음악 표현과 이질적 주석으로 구간 경계·기능 분석. | 프레이즈·곡 구조 요소의 후보 기반. 구간 기능과 세부 프레이즈 정렬은 구별. | [요약](notes/04_hao_songformer.md) · [PDF](papers/04_hao_2025_songformer_arxiv_v3.pdf) · `hao2025songformer` |
| 5 | Fang et al. (2024), *Exploring GPT's Ability as a Judge in Music Understanding*. ISMIR 2024. | 기호 음악 입력을 이용한 박자·코드·조성 주석 오류 검출. | 인식 결과를 구조화해 LLM에 전달하는 설계 근거. 소형 Qwen 전이는 별도 평가. | [요약](notes/05_fang_gpt_music_judge.md) · [PDF](papers/05_fang_2024_gpt_music_judge_ismir.pdf) · `fang2024gptjudge` |
| 6 | Koh et al. (2020), *Concept Bottleneck Models*. ICML 2020. | 사람이 이해하는 개념을 거쳐 예측하며 개념 개입을 지원. | 음원 → 다섯 요소 → 판단 구조와 개입 평가의 이론적 기반. | [요약](notes/06_koh_concept_bottleneck.md) · [PDF](papers/06_koh_2020_concept_bottleneck_icml.pdf) · `koh2020concept` |
| 7 | Turpin et al. (2023), *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting*. NeurIPS 2023. | 자연어 설명이 판단에 작용한 요인을 충실히 드러내지 못할 수 있음을 실험. | 설명 충실도 평가의 동기. 근거 제거·변경·순서 교환 등은 우리 연구에 맞춘 응용. | [요약](notes/07_turpin_unfaithful_explanations.md) · [PDF](papers/07_turpin_2023_unfaithful_explanations_neurips.pdf) · `turpin2023unfaithful` |
| 8 | Wang et al. (2020), *POP909: A Pop-song Dataset for Music Arrangement Generation*. ISMIR 2020. | 909곡의 피아노 편곡 MIDI와 선율·반주 및 음악 주석 제공. | 통제 변형 쌍 제작의 기반. 표절·실제 베이스 연주 정답은 제공하지 않음. | [요약](notes/08_wang_pop909.md) · [PDF](papers/08_wang_2020_pop909_ismir.pdf) · `wang2020pop909` |

## 서지정보를 읽을 때

- **1번:** 학회 연도는 2024, arXiv 제출일은 2026-01-05이다. 학회 문헌을 인용하는 항목으로 작성하고 arXiv 공개 시점을 별도 기록했다. [공식 기록](https://arxiv.org/abs/2601.02586)
- **3번:** 학회 공식 PDF를 저장했다. arXiv 최초 공개는 2023년이고, 확인 시 arXiv 최신판은 v5(2024-12-27)로 학회 PDF와 구별된다. [학회 기록](https://proceedings.iclr.cc/paper_files/paper/2024/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html) · [arXiv 기록](https://arxiv.org/abs/2306.00107)
- **4번:** 최초 공개 2025-10-03, 저장 판본 v3의 개정일은 2026-04-08이다. 확인한 arXiv 기록을 기준으로 프리프린트로 표기했다. [공식 기록](https://arxiv.org/abs/2510.02797)

## 권장 읽기 순서

1. **1 → 2:** 연구 질문과 인간의 요소별 유사성 평가.
2. **3 → 4:** 음악 표현과 곡 구조 분석.
3. **5 → 6 → 7:** LLM 입력 설계, 개념 기반 판단, 설명의 충실도.
4. **8:** 통제된 데이터 제작과 평가 분할.

음원·데이터셋·모델 가중치는 이 폴더에 포함하지 않았다. PDF의 실제 저장 여부와 판본은 `sources.json`을 기준으로 확인한다.

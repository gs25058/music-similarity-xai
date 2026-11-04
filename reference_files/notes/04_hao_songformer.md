# 04. SongFormer 곡 구조 분석

- **저자:** Chunbo Hao, Ruibin Yuan, Jixun Yao, Qixin Deng, Xinyi Bai, Yanbo Wang, Wei Xue, Lei Xie.
- **제목:** SongFormer: Scaling Music Structure Analysis with Heterogeneous Supervision.
- **발표:** arXiv 프리프린트. 최초 공개 2025-10-03; 저장 판본 v3, 2026-04-08 개정.
- **출처:** [공식 서지·초록](https://arxiv.org/abs/2510.02797) · [고정 판본](https://arxiv.org/abs/2510.02797v3).
- **인용 키:** `hao2025songformer`.

## 확인한 내용

짧은 창과 긴 창의 자기지도 음악 표현을 결합하고, 출처 임베딩으로 부분적이거나 서로 다른 체계의 주석을 활용한다. 곡 구조의 경계와 기능을 분석하며 SongFormDB와 SongFormBench를 제시한다. 근거 범위: 공식 초록과 버전 이력.

## 우리 연구의 활용 제안

다섯 번째 요소인 프레이즈·곡 구조 분석의 후보 기반이다. 유사 구간의 시작·끝과 verse/chorus 등의 기능을 설명에 포함하는 데 활용할 수 있다. 구조 경계와 기능 라벨은 각각 평가하고, 구조 모듈을 제거한 경우와 비교한다.

## 한계와 후속 확인

곡의 큰 구간 경계·기능을 예측하는 능력과 세밀한 프레이즈 정렬 능력은 구분해야 한다. 이 모델의 출력이 곧 음악적 유사성이나 표절 판단은 아니다. 프레이즈 정의, 추론 입력 길이, 모델·데이터의 실제 이용 조건은 별도 확인한다. 2025년 최초판과 2026년 개정판의 결과를 섞어 인용하지 않는다.

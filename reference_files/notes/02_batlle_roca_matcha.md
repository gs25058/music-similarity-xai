# 02. 음악 속성별 인간·계산 판단의 정렬과 MATCHA

- **저자:** Roser Batlle-Roca, Woosung Choi, Joan Serrà, Fabio Morreale, Wei-Hsiang Liao, Xavier Serra, Emilia Gómez, Yuki Mitsufuji.
- **제목:** On the Human and Computer Alignment of Attribute-Based Music Matches.
- **발표:** arXiv 프리프린트, v1, 2026-09-01.
- **출처:** [공식 서지·초록](https://arxiv.org/abs/2609.00987) · [고정 판본](https://arxiv.org/abs/2609.00987v1).
- **인용 키:** `batlleroca2026matches`.

## 확인한 내용

선율·화성·리듬·목소리·음색의 다섯 속성에서 강하게 유사한 음악 발췌를 평가한다. 300개 사례의 삼중항 강제선택 과제를 구성했으며, MATCHA는 전문가 83명의 지각 평가 1,105건을 담는다. 인간 판단과 계산 지표 사이의 정렬은 부분적이다. 근거 범위: 공식 초록.

## 우리 연구의 활용 제안

전체 유사도 외에 요소별 인간 판단을 수집하고 모델 점수와 비교하는 평가 설계의 근거로 사용한다. 삼중항 선택 과제와 요소별 점수 평가를 후보로 두되, 지표와 참여자 조건은 별도로 설계한다.

## 한계와 매핑

이 논문의 목소리·음색을 우리의 반주 패턴·구조와 같은 개념으로 취급하지 않는다. 선율과 음정 관계도 동일한 정의로 가정하지 않는다. MATCHA를 사용하려면 실제 공개 데이터의 스키마·접근 조건·분할을 확인하고, 공통 속성에 대한 평가부터 검토한다. 인간이 유사하다고 고른 결과를 표절 정답으로 전용하지 않는다.

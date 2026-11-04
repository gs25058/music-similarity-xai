# 06. Concept Bottleneck Models

- **저자:** Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, Percy Liang.
- **발표:** ICML 2020, PMLR 119, pp. 5338–5348.
- **출처:** [공식 논문 페이지](https://proceedings.mlr.press/v119/koh20a.html) · [공식 PDF](https://proceedings.mlr.press/v119/koh20a/koh20a.pdf).
- **인용 키:** `koh2020concept`.

## 확인한 내용

입력에서 사람이 이해하는 개념을 예측한 뒤 그 개념으로 최종 라벨을 예측한다. 예측된 개념 값을 수정해 최종 판단에 개입할 수 있다. 음악 분야를 직접 검증한 연구는 아니다. 근거 범위: 공식 초록.

## 우리 연구의 활용 제안

음원 → 다섯 음악 요소 → 최종 판단의 구조적 근거로 사용한다. 예를 들어 리듬 근거만 교체하고 다른 근거를 고정한 뒤 판단과 설명의 변화를 측정하는 개입 평가를 설계한다.

## 설계상 구분

다섯 요소를 설명에 나열하는 것만으로 개념 병목 모델이 되는 것은 아니다. 최종 판단이 어떤 개념 입력에 의존하는지 정의하고, 숨은 음원 임베딩이나 별도 입력이 판단으로 우회하는지 확인해야 한다. 완전한 병목을 구현하지 않는 경우에는 ‘개념 병목에서 착안한 구조’라고 기술한다. 개념의 정확도와 개입 효과도 별도로 평가한다.

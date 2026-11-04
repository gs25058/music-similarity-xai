# 07. LLM 설명의 비충실성

- **저자:** Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman.
- **제목:** Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.
- **발표:** NeurIPS 2023.
- **출처:** [공식 논문 페이지](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract-Conference.html) · [공식 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/ed3fea9033a80fea1376299fa7863f4a-Paper-Conference.pdf).
- **인용 키:** `turpin2023unfaithful`.

## 확인한 내용

입력의 편향 요인이 모델의 답변에 영향을 주어도, 단계별 자연어 설명이 그 영향을 드러내지 않을 수 있음을 실험한다. 설명의 그럴듯함만으로 판단 근거의 충실성을 보장할 수 없다. 음악 판단을 직접 실험한 논문은 아니다. 근거 범위: 공식 초록.

## 우리 연구의 활용 제안

설명의 읽기 쉬움과 근거 충실도를 분리해 평가한다. 아래는 이 논문을 동기로 우리 과제에 맞춰 설계할 검사이며, 원 논문의 음악 실험 결과가 아니다.

| 검사 | 변경 | 관찰할 항목 |
| --- | --- | --- |
| 근거 제거 | 특정 요소의 근거를 누락 처리 | 그 요소에 대한 단정이 줄어드는가 |
| 근거 변경 | 특정 요소의 점수·구간을 일관되게 교체 | 판단·설명이 바뀐 근거에 반응하는가 |
| 순서 교환 | 같은 곡 쌍의 A/B 순서를 교환 | 대칭적 유사도와 역할 표현이 일관되는가 |
| 입력 근거 대조 | 출력 주장을 입력과 대조 | 입력에 없는 사실을 만들어내는가 |

누락된 근거를 유사도 0으로 대체하지 않는다. 이러한 행동 검사가 내부 계산 과정을 완전히 증명하는 것은 아니다.

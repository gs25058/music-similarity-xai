# 03. MERT 음악 표현 모델

- **저자:** Yizhi Li, Ruibin Yuan, Ge Zhang, Yinghao Ma, Xingran Chen, Hanzhi Yin, Chenghao Xiao, Chenghua Lin, Anton Ragni, Emmanouil Benetos, Norbert Gyenge, Roger Dannenberg, Ruibo Liu, Wenhu Chen, Gus Xia, Yemin Shi, Wenhao Huang, Zili Wang, Yike Guo, Jie Fu.
- **제목:** MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training.
- **발표:** ICLR 2024. 저장 PDF는 학회 공식판. arXiv 최초 공개는 2023년이며 최신 확인판은 v5(2024-12-27).
- **출처:** [학회 공식 페이지](https://proceedings.iclr.cc/paper_files/paper/2024/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2306.00107).
- **인용 키:** `li2024mert`.

## 확인한 내용

음향 교사와 음고 정보를 반영하는 CQT 기반 음악 교사를 이용해 자기지도 음악 표현을 학습한다. 여러 음악 이해 과제에서 표현을 평가한다. 학회 원문 §4.1의 probing 평가에서는 사전학습 백본을 고정하고 간단한 downstream 모델을 학습한다.

## 우리 연구의 활용 제안

음원 인코더의 후보 기반으로 사용한다. 음정·리듬·화성·반주 패턴별 유사성 헤드를 별도 학습하고, 백본 고정 기준선과 일부 미세조정 방식을 비교한다. 요소별 민감도는 특정 요소만 바꾼 통제 쌍으로 평가한다.

## 한계와 구분

범용 음악 표현의 성능이 다섯 요소의 분리나 음악 표절 판단 성능을 보장하지 않는다. 부분 미세조정과 요소별 유사성 학습은 우리 연구의 확장이다. 근거 범위: 공식 초록 및 학회 PDF §4.1. 실제 체크포인트·추론 조건은 구현 전에 별도 확인한다.

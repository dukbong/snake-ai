# Snake AI — PPO 강화학습

Snake 게임을 스스로 학습하는 AI입니다.
11차원 상태 표현 + DQN → CNN + PPO 방식으로 발전시키며 구현했습니다.

![학습 그래프](plots/training_progress.png)

---

## 결과

| 항목 | 값 |
|------|-----|
| 최고기록 | 56 |
| 최근 평균 | 22 ~ 28 |
| 총 학습 스텝 | 6,000,000+ |
| 총 학습 게임 수 | 31,000+  |

---

## 구조

```
snake-ai/
├── game.py            # Snake 게임 환경
├── model.py           # CNN Actor-Critic 네트워크
├── agent.py           # PPO 에이전트 + RolloutBuffer
├── vec_env.py         # 단일 프로세스 벡터화 환경
├── subproc_vec_env.py # 멀티프로세스 벡터화 환경
├── main.py            # 학습 실행
├── play.py            # 학습된 모델 플레이 관람
└── benchmark.py       # 성능 프로파일링
```

---

## 알고리즘

### PPO (Proximal Policy Optimization)
정책을 직접 학습하는 on-policy 알고리즘입니다.
업데이트 크기를 `clip_epsilon=0.2`로 제한해 학습 안정성을 확보합니다.

### CNN Actor-Critic
게임 상태를 7채널 그리드(24×32)로 표현합니다.

| 채널 | 의미 |
|------|------|
| 0 | 뱀 몸통 |
| 1 | 뱀 머리 |
| 2 | 음식 |
| 3~6 | 방향 one-hot |

### GAE (Generalized Advantage Estimation)
단기/장기 보상 추정의 편향-분산 균형을 `λ=0.95`로 조절합니다.

### 보상 함수
| 상황 | 보상 |
|------|------|
| 음식 먹음 | +1.0 |
| 충돌 (사망) | -1.0 |
| 음식에 가까워짐 | +0.01 |
| 음식에서 멀어짐 | -0.01 |

---

## 학습 방법

### 기본 학습
```bash
python main.py --no-render
```

### 이어서 학습 (LR 초기화)
```bash
python main.py --no-render --reset-steps
```

### 주요 옵션
```
--n-envs          병렬 환경 수 (기본: 8)
--n-steps         rollout 길이 (기본: 128)
--total-timesteps 총 학습 스텝 (기본: 3,000,000)
--no-multiprocess 단일 프로세스 모드
--no-render       렌더링 비활성화
```

---

## 학습된 모델 실행
```bash
python play.py          # 무한 반복
python play.py --games 5  # 5게임만 관람
```

---

## 성능 프로파일링
```bash
python benchmark.py
```

---

## 학습 과정

| 학습 | 최고기록 | 평균 |
|------|---------|------|
| 1차 (3M 스텝) | 39 | ~15 |
| 2차 (reset-steps, 3M 스텝) | 56 | ~22~28 |

---

## 환경

- Python 3.12
- PyTorch
- pygame
- numpy

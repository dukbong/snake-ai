# Snake AI — PPO 강화학습 에이전트

Snake 게임을 스스로 학습하는 강화학습 AI입니다.

Stable-Baselines 등 외부 RL 라이브러리 없이 PPO 알고리즘을 구성했습니다.

![학습 그래프](plots/training_progress.png)

---

## 성과

| 항목 | 값 |
|------|-----|
| 최고 점수 | **104** |
| 최근 20게임 평균 | **~50** |
| 총 학습 스텝 | **14,800,000+** |
| 총 학습 게임 수 | **50,000+** |

---

## 기술 스택

`Python` `PyTorch` `NumPy` `pygame` `multiprocessing`

---

## 개발 배경

처음에는 **11차원 상태 표현 + DQN** 방식으로 시작했습니다.

- 11차원 방식: 빠르게 평균 20에 도달했지만 이후 성장이 없었습니다.
- DQN: 느리지만 꾸준히 성장했으나 결국 한계에 부딪혔습니다.

두 방식의 한계를 직접 경험한 후 **CNN 기반 상태 표현 + PPO**로 전환했고 지속적인 성장을 확인했습니다.

---

## 구현 내용

### PPO (Proximal Policy Optimization)

- **Clipped Surrogate Objective**: `clip_epsilon=0.2`로 정책 업데이트 크기를 제한해 학습 안정성 확보
- **GAE** (Generalized Advantage Estimation): `λ=0.95`로 편향-분산 균형 조절
- **Entropy Bonus**: 스테이지별 차등 엔트로피 계수, cosine 감쇠 적용
- **done / truncated 분리**: 타임아웃과 실제 충돌 종료를 구분해 GAE 계산 오류 방지
- **Cosine Annealing**: 스테이지별 독립 cosine cycle (LR warmup + LR floor 보장)
- **GPU/MPS 가속**: CUDA, Apple MPS, CPU 자동 감지

### ResNet Actor-Critic 네트워크 (~1.95M 파라미터)

게임 상태를 **7채널 그리드**로 표현합니다.

| 채널 | 의미 |
|------|------|
| 0 | 뱀 몸통 |
| 1 | 뱀 머리 |
| 2 | 음식 위치 |
| 3 ~ 6 | 현재 방향 (one-hot) |

```
Conv2d(7→128) + GroupNorm → ResBlock(128) ×3 → AdaptiveAvgPool(4,4) → FC(512)
                                                                        ├── Policy Head → 3 actions
                                                                        └── Value Head  → scalar
```

- **ResBlock**: GroupNorm + Conv3×3 + residual connection (RL에서 BN 불안정 방지)
- **AdaptiveAvgPool2d(4,4)**: 임의 그리드 크기를 고정 차원으로 변환 (커리큘럼 호환)
- **Orthogonal 초기화**: 학습 초기 안정성 확보

### 커리큘럼 학습

작은 그리드에서 시작해 점진적으로 풀 사이즈까지 확장합니다.

| 단계 | 그리드 | 셀 수 | 승급 조건 (최근 500게임 평균) |
|------|--------|-------|------------------------------|
| 0 | 8×8 | 64 | ≥ 20 |
| 1 | 12×12 | 144 | ≥ 40 |
| 2 | 16×16 | 256 | ≥ 60 |
| 3 | 20×24 | 480 | ≥ 90 |
| 4 | 24×32 | 768 | 최종 |

- **편도 승급**: 강등 없음
- **Cosine Annealing**: 각 스테이지마다 독립적인 cosine cycle (최소 10M 스텝)
- **Value Warm-up**: 승급 직후 value_coeff를 점진 복구 (0.1 → 0.5)
- **타임아웃 스케일링**: 그리드 면적에 비례하여 타임아웃 조정

### 멀티프로세스 병렬 환경

`multiprocessing.Process` + `Pipe` 기반 `SubprocVecEnv`를 구현했습니다.

- 32개 환경이 각각 독립 프로세스에서 동시 실행
- `forkserver` 방식으로 macOS MPS 호환성 확보
- 커리큘럼 승급 시 `resize` 명령으로 그리드 크기 동적 변경

### 보상 함수

| 상황 | 보상 |
|------|------|
| 음식 획득 | +1.0 |
| 충돌 (사망) | -1.0 |
| 그 외 | 0 |

Sparse reward 구조를 유지합니다. 거리 기반 보상은 사용하지 않습니다.

---

## 학습 성장 과정

| 학습 회차 | 방식 | 최고기록 | 평균 |
|---------|------|---------|------|
| 1차 (3M 스텝) | PPO, n_steps=128 | 39 | ~15 |
| 2차 (3M 스텝) | reset-steps | 56 | ~26 |
| 3차 (3M 스텝) | reset-steps, n_steps=256 | 64 | ~30 |
| 4차 (3M 스텝) | reset-steps, entropy annealing 도입 | 87 | ~35 |
| 5차 (3M 스텝) | reset-steps | **104** | **~50** |

---

## 실행 방법

### 설치
```bash
pip install torch pygame numpy matplotlib
```

### 학습된 모델 실행 (관람)
```bash
python play.py              # 무한 반복
python play.py --games 5    # 5게임 관람
python play.py --grid 8x8   # 특정 그리드 크기로 관람
```

### 학습
```bash
# 기본 학습 (커리큘럼 + GPU/MPS 자동 감지)
python main.py --no-render

# 체크포인트에서 이어서 학습 (스텝 카운터 초기화)
python main.py --no-render --reset-steps

# 커리큘럼 비활성화 (24×32 고정)
python main.py --no-render --no-curriculum

# 커리큘럼 임계값 조정
python main.py --no-render --curriculum-thresholds 30,50,70,100

# 디바이스 명시 지정
python main.py --no-render --device cuda

# 단일 프로세스 모드 (렌더링 포함)
python main.py --no-multiprocess
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--n-envs` | 32 | 병렬 환경 수 |
| `--n-steps` | 256 | rollout 길이 |
| `--total-timesteps` | 50,000,000 | 총 학습 스텝 |
| `--lr` | 1e-4 | 학습률 |
| `--entropy-coeff` | 0.05 | 엔트로피 계수 시작값 |
| `--device` | auto | 디바이스 (cuda/mps/cpu) |
| `--curriculum-thresholds` | 20,40,60,90 | 커리큘럼 승급 임계값 |
| `--frame-stack` | 1 | 프레임 스태킹 수 (1 = 비활성화) |
| `--no-curriculum` | - | 커리큘럼 비활성화 |

---

## 프로젝트 구조

```
snake-ai/
├── game.py             # Snake 게임 환경 (pygame), 가변 그리드 크기 지원
├── model.py            # ResNet Actor-Critic 네트워크 (ResBlock + GroupNorm)
├── agent.py            # PPO 에이전트 + RolloutBuffer + GAE + GPU 지원
├── vec_env.py          # 단일 프로세스 벡터화 환경
├── subproc_vec_env.py  # 멀티프로세스 벡터화 환경 (forkserver, resize 지원)
├── main.py             # 학습 루프 (커리큘럼, cosine annealing, value warm-up)
├── play.py             # 학습된 모델 플레이 관람
└── benchmark.py        # 성능 프로파일링
```

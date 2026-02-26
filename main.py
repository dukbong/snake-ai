import argparse
import math
from collections import deque
import numpy as np
import torch
from agent import PPOAgent
from vec_env import VecSnakeEnv
from subproc_vec_env import SubprocVecEnv
from helper import plot

CHECKPOINT_FILE = 'model_ppo.pth'

# 커리큘럼 스케줄: (rows, cols)
CURRICULUM_GRIDS = [
    (8, 8),      # Stage 0: 64 셀
    (12, 12),    # Stage 1: 144 셀
    (16, 16),    # Stage 2: 256 셀
    (20, 24),    # Stage 3: 480 셀
    (24, 32),    # Stage 4: 768 셀 (최종)
]

# 스테이지별 엔트로피 (시작값, 종착값) — cosine 감쇠 적용
STAGE_ENTROPY = {
    0: (0.05, 0.02),
    1: (0.04, 0.015),
    2: (0.05, 0.02),       # 16×16 premature convergence 방지
    3: (0.025, 0.008),
    4: (0.02, 0.005),
}

# Value warm-up: 승급 후 value_coeff를 점진 복구
VALUE_WARMUP_UPDATES = 50
VALUE_WARMUP_START = 0.1
VALUE_WARMUP_END = 0.5

# LR warmup: 각 cosine cycle 시작 시 처음 N gradient steps 동안 linear warmup
LR_WARMUP_STEPS = 500

# 각 스테이지 최소 cosine cycle 길이 (스텝 수)
MIN_CYCLE_STEPS = 10_000_000


def train():
    parser = argparse.ArgumentParser(description='Snake AI PPO 학습')
    parser.add_argument('--no-render', action='store_true',
                        help='렌더링 없이 학습 (UI 비활성화로 빠른 학습)')
    parser.add_argument('--n-envs', type=int, default=32,
                        help='다중 환경 수 (기본: 32)')
    parser.add_argument('--n-steps', type=int, default=256,
                        help='rollout 길이 (기본: 256)')
    parser.add_argument('--total-timesteps', type=int, default=50_000_000,
                        help='총 학습 스텝 (기본: 50,000,000)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='기본 학습률 (기본: 1e-4)')
    parser.add_argument('--entropy-coeff', type=float, default=0.05,
                        help='엔트로피 계수 시작값 (기본: 0.05)')
    parser.add_argument('--reset-steps', action='store_true',
                        help='스텝 카운터 초기화 후 학습 재시작')
    parser.add_argument('--no-multiprocess', action='store_true',
                        help='멀티프로세스 비활성화 (단일 프로세스 VecSnakeEnv 사용)')
    parser.add_argument('--device', type=str, default=None,
                        help='디바이스 지정 (cuda/mps/cpu, 기본: auto-detect)')
    parser.add_argument('--curriculum', action='store_true', default=True,
                        help='커리큘럼 학습 활성화 (기본: True)')
    parser.add_argument('--no-curriculum', action='store_true',
                        help='커리큘럼 학습 비활성화 (24×32 고정)')
    parser.add_argument('--curriculum-thresholds', type=str, default='20,40,60,90',
                        help='커리큘럼 승급 임계값 (쉼표 구분, 기본: 20,40,60,90)')
    parser.add_argument('--frame-stack', type=int, default=1,
                        help='프레임 스태킹 수 (기본: 1 = 비활성화)')
    args = parser.parse_args()

    use_curriculum = args.curriculum and not args.no_curriculum
    curriculum_thresholds = [int(x) for x in args.curriculum_thresholds.split(',')]
    frame_stack_n = args.frame_stack

    n_envs = args.n_envs
    n_steps = args.n_steps
    total_timesteps = args.total_timesteps
    base_lr = args.lr
    steps_per_update = n_steps * n_envs

    # 커리큘럼 상태
    curriculum_stage = 0
    if use_curriculum:
        grid_rows, grid_cols = CURRICULUM_GRIDS[curriculum_stage]
    else:
        grid_rows, grid_cols = 24, 32
        curriculum_stage = len(CURRICULUM_GRIDS) - 1  # 최종 스테이지

    # in_channels: frame_stack 적용 시 공간 채널(3) * N + 방향(4)
    in_channels = 3 * frame_stack_n + 4 if frame_stack_n > 1 else 7
    obs_shape = (in_channels, grid_rows, grid_cols)

    # 생성 순서: SubprocVecEnv → PPOAgent (worker가 PyTorch 초기화 이전에 생성되어야 함)
    if not args.no_multiprocess:
        if not args.no_render:
            print("멀티프로세스 모드: 렌더링 비활성화 (--no-multiprocess로 단일 프로세스 전환 가능)")
        vec_env = SubprocVecEnv(n_envs=n_envs, grid_rows=grid_rows, grid_cols=grid_cols)
    else:
        vec_env = VecSnakeEnv(n_envs=n_envs, render=not args.no_render,
                              grid_rows=grid_rows, grid_cols=grid_cols)

    agent = PPOAgent(
        n_envs=n_envs, n_steps=n_steps, lr=base_lr,
        entropy_coeff=args.entropy_coeff, in_channels=in_channels,
        obs_shape=obs_shape, device=args.device,
    )

    # 체크포인트 자동 로드 (이어서 학습)
    checkpoint = agent.load_checkpoint(CHECKPOINT_FILE)
    if checkpoint is not None:
        # 커리큘럼 상태 복원
        curriculum_stage = checkpoint.get('curriculum_stage', 0)
        saved_scores = checkpoint.get('curriculum_scores', [])
        cycle_step_counter = checkpoint.get('cycle_step_counter', 0)
        value_warmup_counter = checkpoint.get('value_warmup_counter', VALUE_WARMUP_UPDATES)
        grad_step_counter = checkpoint.get('grad_step_counter', LR_WARMUP_STEPS)

        if use_curriculum:
            grid_rows, grid_cols = CURRICULUM_GRIDS[curriculum_stage]
            obs_shape = (in_channels, grid_rows, grid_cols)
            # 환경 리사이즈 (체크포인트 스테이지가 0이 아닌 경우)
            if curriculum_stage > 0:
                vec_env.set_grid_size(grid_rows, grid_cols)
            agent.buffer.reallocate(obs_shape)

        if args.reset_steps:
            agent.total_steps = 0
            cycle_step_counter = 0
            grad_step_counter = 0
            saved_scores = []          # 기존 점수 데이터가 승급 평균 끌어내리는 것 방지
            print(f"스텝 초기화: lr={base_lr:.2e}")
    else:
        saved_scores = []
        cycle_step_counter = 0
        value_warmup_counter = VALUE_WARMUP_UPDATES  # 초기에는 warm-up 불필요 (이미 완료 상태)
        grad_step_counter = 0
        print("Snake AI PPO 학습 시작!")

    mode_str = "단일 프로세스" if args.no_multiprocess else "멀티프로세스"
    if args.no_render or not args.no_multiprocess:
        print(f"렌더링 비활성화: 빠른 학습 모드 ({mode_str})")
    print(f"Device: {agent.device}")
    print(f"n_envs={n_envs}, n_steps={n_steps}, 배치크기={steps_per_update}, lr={base_lr:.2e}")
    if use_curriculum:
        grid_str = f"{grid_rows}×{grid_cols}"
        print(f"커리큘럼: Stage {curriculum_stage} ({grid_str}), 임계값={curriculum_thresholds}")
    else:
        print("커리큘럼 비활성화: 24×32 고정")
    print("Ctrl+C로 중단 시 자동 저장됩니다.")
    print("-" * 70)

    # 승급 판단용 점수 기록 (최근 500게임)
    episode_scores = deque(saved_scores, maxlen=500)
    plot_scores = []
    plot_mean_scores = []
    promotion_pending = False

    # Cosine cycle: 스테이지별 독립 cycle
    lr_floor = base_lr * 0.05
    cycle_total_steps = MIN_CYCLE_STEPS

    states = vec_env.reset()
    dones = np.zeros(n_envs, dtype=bool)

    # while 루프: auto-extend 지원
    update_idx = 0
    max_steps = total_timesteps

    try:
        while agent.total_steps < max_steps:
            new_record = False

            # --- LR & ENTROPY SCHEDULING: Cosine Annealing (스테이지별 독립 cycle) ---
            cycle_frac = min(cycle_step_counter / cycle_total_steps, 1.0)

            # LR warmup: 첫 N gradient steps 동안 linear warmup
            if grad_step_counter < LR_WARMUP_STEPS:
                warmup_frac = grad_step_counter / LR_WARMUP_STEPS
                lr_now = lr_floor + (base_lr - lr_floor) * warmup_frac
            else:
                # Cosine annealing: base_lr → lr_floor
                cosine_decay = 0.5 * (1 + math.cos(math.pi * cycle_frac))
                lr_now = lr_floor + (base_lr - lr_floor) * cosine_decay

            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = lr_now

            # 엔트로피: 스테이지별 차등 (시작값, 종착값) → cosine 감쇠
            ent_start, ent_end = STAGE_ENTROPY.get(curriculum_stage, (0.02, 0.005))
            ent_decay = 0.5 * (1 + math.cos(math.pi * cycle_frac))
            ent_coeff_now = ent_end + (ent_start - ent_end) * ent_decay

            # Value warm-up: 승급 후 value_coeff 점진 복구
            if value_warmup_counter < VALUE_WARMUP_UPDATES:
                effective_value_coeff = VALUE_WARMUP_START + \
                    (VALUE_WARMUP_END - VALUE_WARMUP_START) * value_warmup_counter / VALUE_WARMUP_UPDATES
                value_warmup_counter += 1
            else:
                effective_value_coeff = VALUE_WARMUP_END

            # --- ROLLOUT 수집 ---
            for step in range(n_steps):
                actions, log_probs, values = agent.get_action(states)
                next_states, rewards, dones, truncateds, infos = vec_env.step(actions)

                # truncated 환경: terminal_obs로 V(s_terminal) 계산
                terminal_values = np.zeros(n_envs, dtype=np.float32)
                for i in range(n_envs):
                    if 'terminal_obs' in infos[i]:
                        t_obs = torch.from_numpy(infos[i]['terminal_obs']).unsqueeze(0).to(agent.device)
                        with torch.no_grad():
                            terminal_values[i] = agent.model.get_value(t_obs).cpu().item()

                agent.buffer.store(
                    step, states, actions, log_probs, rewards, values,
                    dones, truncateds, terminal_values
                )
                states = next_states

                # 에피소드 완료 로깅
                for i in range(n_envs):
                    if 'terminal_score' in infos[i]:
                        score = infos[i]['terminal_score']
                        episode_scores.append(score)
                        agent.n_games += 1
                        if score > agent.record:
                            agent.record = score
                            new_record = True

                        # 커리큘럼 승급 판단 (롤아웃 도중에는 플래그만 세움)
                        if use_curriculum and not promotion_pending:
                            final_stage = len(CURRICULUM_GRIDS) - 1
                            if curriculum_stage < final_stage and len(episode_scores) >= 500:
                                avg_score = sum(episode_scores) / len(episode_scores)
                                threshold = curriculum_thresholds[curriculum_stage] \
                                    if curriculum_stage < len(curriculum_thresholds) else float('inf')
                                if avg_score >= threshold:
                                    promotion_pending = True

            agent.total_steps += steps_per_update
            cycle_step_counter += steps_per_update

            # --- BOOTSTRAP: 마지막 상태의 value 계산 ---
            with torch.no_grad():
                last_values = agent.model.get_value(
                    torch.from_numpy(states).to(agent.device)
                ).cpu().numpy()
            last_dones = dones

            # --- UPDATE ---
            metrics = agent.update(last_values, last_dones,
                                   entropy_coeff=ent_coeff_now,
                                   value_coeff=effective_value_coeff)

            # gradient step 카운터 업데이트
            # 대략적인 gradient steps = n_epochs * ceil(total_samples / batch_size)
            total_samples = n_steps * n_envs
            grad_steps_this_update = agent.n_epochs * math.ceil(total_samples / agent.batch_size)
            grad_step_counter += grad_steps_this_update

            update_idx += 1

            # --- 커리큘럼 승급 실행 (롤아웃 경계에서만) ---
            if promotion_pending:
                curriculum_stage += 1
                grid_rows, grid_cols = CURRICULUM_GRIDS[curriculum_stage]
                obs_shape = (in_channels, grid_rows, grid_cols)

                # a. worker resize → 새 초기 상태 수신
                states = vec_env.set_grid_size(grid_rows, grid_cols)
                dones = np.zeros(n_envs, dtype=bool)

                # b. RolloutBuffer 재할당
                agent.buffer.reallocate(obs_shape)

                # c. Cosine cycle 리셋
                cycle_step_counter = 0
                cycle_total_steps = MIN_CYCLE_STEPS
                # auto-extend: 현재 스테이지 min cycle이 남은 예산보다 크면 연장
                if agent.total_steps + MIN_CYCLE_STEPS > max_steps:
                    max_steps = agent.total_steps + MIN_CYCLE_STEPS

                # d. LR warmup 리셋
                grad_step_counter = 0

                # e. Value warm-up 시작
                value_warmup_counter = 0

                # f. 승급 플래그 해제
                promotion_pending = False

                grid_str = f"{grid_rows}×{grid_cols}"
                avg = sum(episode_scores) / len(episode_scores) if episode_scores else 0
                print(f"\n{'='*70}")
                print(f"[커리큘럼 승급] Stage {curriculum_stage} ({grid_str}) | "
                      f"평균 점수: {avg:.1f} | 게임: {agent.n_games}")
                print(f"max_steps 연장: {max_steps:,} | cycle: {cycle_total_steps:,}")
                print(f"{'='*70}\n")

                # g. 이전 스테이지 점수 초기화
                episode_scores.clear()

            # --- LOGGING (10 업데이트마다) ---
            if update_idx % 10 == 0 and episode_scores:
                recent = list(episode_scores)[-20:]
                recent_avg = sum(recent) / len(recent)
                plot_scores.append(recent[-1])
                plot_mean_scores.append(round(recent_avg, 2))

                grid_str = f"{grid_rows}×{grid_cols}"
                print(
                    f"업데이트: {update_idx:5d} | "
                    f"스텝: {agent.total_steps:10d} | "
                    f"게임: {agent.n_games:5d} | "
                    f"최근평균: {recent_avg:.2f} | "
                    f"최고: {agent.record:3d} | "
                    f"policy: {metrics['policy_loss']:.4f} | "
                    f"value: {metrics['value_loss']:.4f} | "
                    f"ent: {metrics['entropy']:.4f} | "
                    f"KL: {metrics['approx_kl']:.4f} | "
                    f"lr: {lr_now:.2e} | "
                    f"ent_c: {ent_coeff_now:.4f} | "
                    f"v_c: {effective_value_coeff:.3f} | "
                    f"S{curriculum_stage}({grid_str})"
                )
                plot(plot_scores, plot_mean_scores)

            # --- CHECKPOINT ---
            if new_record:
                agent.save_checkpoint(CHECKPOINT_FILE, extra={
                    'curriculum_stage': curriculum_stage,
                    'curriculum_scores': list(episode_scores),
                    'cycle_step_counter': cycle_step_counter,
                    'value_warmup_counter': value_warmup_counter,
                    'grad_step_counter': grad_step_counter,
                    'frame_stack_n': frame_stack_n,
                })
            elif update_idx % 50 == 0:
                agent.save_checkpoint(CHECKPOINT_FILE, extra={
                    'curriculum_stage': curriculum_stage,
                    'curriculum_scores': list(episode_scores),
                    'cycle_step_counter': cycle_step_counter,
                    'value_warmup_counter': value_warmup_counter,
                    'grad_step_counter': grad_step_counter,
                    'frame_stack_n': frame_stack_n,
                })

    except KeyboardInterrupt:
        print("\n학습 중단...")
    finally:
        agent.save_checkpoint(CHECKPOINT_FILE, extra={
            'curriculum_stage': curriculum_stage,
            'curriculum_scores': list(episode_scores),
            'cycle_step_counter': cycle_step_counter,
            'value_warmup_counter': value_warmup_counter,
            'grad_step_counter': grad_step_counter,
            'frame_stack_n': frame_stack_n,
        })
        vec_env.close()


if __name__ == '__main__':
    train()

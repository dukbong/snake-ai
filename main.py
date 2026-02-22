import argparse
import numpy as np
import torch
from agent import PPOAgent
from vec_env import VecSnakeEnv
from subproc_vec_env import SubprocVecEnv
from helper import plot

CHECKPOINT_FILE = 'model_ppo.pth'


def train():
    parser = argparse.ArgumentParser(description='Snake AI PPO 학습')
    parser.add_argument('--no-render', action='store_true',
                        help='렌더링 없이 학습 (UI 비활성화로 빠른 학습)')
    parser.add_argument('--n-envs', type=int, default=8,
                        help='다중 환경 수 (기본: 8)')
    parser.add_argument('--n-steps', type=int, default=128,
                        help='rollout 길이 (기본: 128)')
    parser.add_argument('--total-timesteps', type=int, default=3_000_000,
                        help='총 학습 스텝 (기본: 3,000,000)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='기본 학습률 (기본: 1e-4)')
    parser.add_argument('--entropy-coeff', type=float, default=0.08,
                        help='엔트로피 계수 (기본: 0.08)')
    parser.add_argument('--reset-steps', action='store_true',
                        help='스텝 카운터 초기화 후 LR annealing 재시작')
    parser.add_argument('--no-multiprocess', action='store_true',
                        help='멀티프로세스 비활성화 (단일 프로세스 VecSnakeEnv 사용)')
    args = parser.parse_args()

    n_envs = args.n_envs
    n_steps = args.n_steps
    total_timesteps = args.total_timesteps
    num_updates = total_timesteps // (n_steps * n_envs)

    # 생성 순서: SubprocVecEnv → PPOAgent (fork가 PyTorch 초기화 이전에 발생해야 함)
    if not args.no_multiprocess:
        if not args.no_render:
            print("멀티프로세스 모드: 렌더링 비활성화 (--no-multiprocess로 단일 프로세스 전환 가능)")
        vec_env = SubprocVecEnv(n_envs=n_envs)
    else:
        vec_env = VecSnakeEnv(n_envs=n_envs, render=not args.no_render)

    agent = PPOAgent(n_envs=n_envs, n_steps=n_steps, lr=args.lr, entropy_coeff=args.entropy_coeff)

    # 체크포인트 자동 로드 (이어서 학습)
    loaded = agent.load_checkpoint(CHECKPOINT_FILE)
    if args.reset_steps and loaded:
        agent.total_steps = 0
        print(f"스텝 초기화: lr={args.lr:.2e}, entropy_coeff={args.entropy_coeff}")
    start_update = agent.total_steps // (n_steps * n_envs)

    if not loaded:
        print("Snake AI PPO 학습 시작!")
    mode_str = "단일 프로세스" if args.no_multiprocess else "멀티프로세스"
    if args.no_render or not args.no_multiprocess:
        print(f"렌더링 비활성화: 빠른 학습 모드 ({mode_str})")
    print(f"n_envs={n_envs}, n_steps={n_steps}, 배치크기={n_envs * n_steps}, 총 업데이트={num_updates}")
    print("Ctrl+C로 중단 시 자동 저장됩니다.")
    print("-" * 70)

    plot_scores = []
    plot_mean_scores = []
    episode_scores = []

    states = vec_env.reset()
    dones = np.zeros(n_envs, dtype=bool)

    try:
        for update_idx in range(start_update, num_updates):
            new_record = False

            # 1. LR ANNEALING: linear decay
            frac = 1.0 - update_idx / num_updates
            lr_now = agent.lr * frac
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = lr_now

            # 2. ROLLOUT 수집
            for step in range(n_steps):
                actions, log_probs, values = agent.get_action(states)
                next_states, rewards, dones, truncateds, infos = vec_env.step(actions)

                # truncated 환경: terminal_obs로 V(s_terminal) 계산
                terminal_values = np.zeros(n_envs, dtype=np.float32)
                for i in range(n_envs):
                    # 'terminal_obs' in infos[i] 으로 체크 (get()으로 numpy 배열을 bool 평가하면 ValueError)
                    if 'terminal_obs' in infos[i]:
                        t_obs = torch.from_numpy(infos[i]['terminal_obs']).unsqueeze(0)
                        with torch.no_grad():
                            terminal_values[i] = agent.model.get_value(t_obs).item()

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

            agent.total_steps += n_steps * n_envs

            # 3. BOOTSTRAP: 마지막 상태의 value 계산
            # auto-reset으로 states는 새 에피소드 obs일 수 있지만,
            # last_dones가 True이면 GAE가 올바르게 bootstrap을 차단함
            with torch.no_grad():
                last_values = agent.model.get_value(torch.from_numpy(states)).numpy()
            last_dones = dones

            # 4. UPDATE
            metrics = agent.update(last_values, last_dones)

            # 5. LOGGING (10 업데이트마다)
            if (update_idx + 1) % 10 == 0 and episode_scores:
                recent = episode_scores[-20:]
                recent_avg = sum(recent) / len(recent)
                plot_scores.append(episode_scores[-1])
                plot_mean_scores.append(round(recent_avg, 2))

                print(
                    f"업데이트: {update_idx+1:5d}/{num_updates} | "
                    f"스텝: {agent.total_steps:8d} | "
                    f"게임: {agent.n_games:5d} | "
                    f"최근평균: {recent_avg:.2f} | "
                    f"최고: {agent.record:3d} | "
                    f"policy: {metrics['policy_loss']:.4f} | "
                    f"value: {metrics['value_loss']:.4f} | "
                    f"ent: {metrics['entropy']:.4f} | "
                    f"KL: {metrics['approx_kl']:.4f} | "
                    f"lr: {lr_now:.2e}"
                )
                plot(plot_scores, plot_mean_scores)

            # 6. CHECKPOINT: 최고기록 갱신 시 즉시 저장, 50 업데이트마다 정기 저장
            if new_record:
                agent.save_checkpoint(CHECKPOINT_FILE)
            elif (update_idx + 1) % 50 == 0:
                agent.save_checkpoint(CHECKPOINT_FILE)

    except KeyboardInterrupt:
        print("\n학습 중단...")
    finally:
        agent.save_checkpoint(CHECKPOINT_FILE)
        vec_env.close()


if __name__ == '__main__':
    train()

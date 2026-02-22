"""
병목 프로파일링 스크립트.

실행:
    python benchmark.py

판단 기준:
    env step이 전체 rollout 시간의 20% 이상 → 멀티프로세스 진행
    10% 미만 → 단일 프로세스 최적화로 충분
"""
import timeit
import numpy as np


def main():
    from game import SnakeGameAI
    from vec_env import VecSnakeEnv
    from agent import PPOAgent

    print("=" * 60)
    print("Snake AI 병목 프로파일링")
    print("=" * 60)

    # (a) 단일 env 128-step rollout
    env = SnakeGameAI(render=False)
    env.reset()

    def single_rollout():
        for _ in range(128):
            _, done, _, _ = env.play_step([1, 0, 0])
            if done:
                env.reset()

    t = timeit.timeit(single_rollout, number=100)
    rollout_ms = t / 100 * 1e3
    print(f"\n[a] 단일 env 128-step rollout: {rollout_ms:.2f} ms")

    # (b) get_grid_state (다양한 snake 길이)
    print("\n[b] get_grid_state 측정:")
    for length in [3, 10, 20]:
        env.reset()
        while len(env.snake) < length:
            env.snake.append(env.snake[-1])
        t = timeit.timeit(lambda: env.get_grid_state(), number=10000)
        print(f"    snake 길이 {length:2d}: {t / 10000 * 1e6:.1f} μs")

    # (c) VecEnv.step vs NN forward
    print("\n[c] VecEnv.step vs NN forward (8 envs):")
    vec_env = VecSnakeEnv(n_envs=8, render=False)
    agent = PPOAgent(n_envs=8)
    states = vec_env.reset()
    actions = np.zeros(8, dtype=int)

    t = timeit.timeit(lambda: vec_env.step(actions), number=1000)
    env_step_ms = t / 1000 * 1e3
    print(f"    VecEnv.step (8 envs): {env_step_ms:.2f} ms")

    t = timeit.timeit(lambda: agent.get_action(states), number=1000)
    nn_ms = t / 1000 * 1e3
    print(f"    NN forward  (8 envs): {nn_ms:.2f} ms")

    # 판단
    total_ms = env_step_ms + nn_ms
    env_ratio = env_step_ms / total_ms * 100 if total_ms > 0 else 0
    print("\n" + "=" * 60)
    print(f"env step 비율: {env_ratio:.1f}% (전체 {total_ms:.2f} ms 중)")
    if env_ratio >= 20:
        print("→ env step이 20% 이상: 멀티프로세스 적용 권장 (3단계 진행)")
    elif env_ratio >= 10:
        print("→ env step이 10~20%: 경계 영역, 멀티프로세스 선택적 적용")
    else:
        print("→ env step이 10% 미만: 단일 프로세스 최적화로 충분")
    print("=" * 60)

    vec_env.close()


if __name__ == '__main__':
    main()

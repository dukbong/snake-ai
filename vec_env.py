import numpy as np
from game import SnakeGameAI

# action 정수 인덱스 → one-hot 변환
ACTION_MAP = [
    [1, 0, 0],  # 0: 직진
    [0, 1, 0],  # 1: 우회전
    [0, 0, 1],  # 2: 좌회전
]


class VecSnakeEnv:
    """N개의 SnakeGameAI 인스턴스를 관리하는 단일 프로세스 래퍼.

    속도 이점은 없으며, rollout 다양성 확보가 목적이다.
    렌더링은 첫 번째 환경(i=0)에만 적용된다.
    """

    def __init__(self, n_envs, render=False):
        self.n_envs = n_envs
        self.envs = []
        for i in range(n_envs):
            env_render = render and (i == 0)
            self.envs.append(SnakeGameAI(render=env_render))

    def reset(self):
        """모든 환경을 리셋하고 초기 상태를 반환한다.

        Returns:
            states: (n_envs, 7, 24, 32) uint8
        """
        states = []
        for env in self.envs:
            env.reset()
            states.append(env.get_grid_state())
        return np.stack(states, axis=0)

    def step(self, actions):
        """모든 환경에서 action을 실행하고 결과를 반환한다.

        Args:
            actions: (n_envs,) int — 정수 인덱스 (0/1/2)

        Returns:
            states:     (n_envs, 7, 24, 32) uint8
            rewards:    (n_envs,) float32 — 보상 정규화 적용 (±10→±1)
            dones:      (n_envs,) bool
            truncateds: (n_envs,) bool
            infos:      list of dict — 'terminal_score', 'terminal_obs' (truncated 시)
        """
        infos = [{} for _ in range(self.n_envs)]  # 매 호출마다 새로 초기화
        states = []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        truncateds = np.zeros(self.n_envs, dtype=bool)

        for i, env in enumerate(self.envs):
            action_onehot = ACTION_MAP[int(actions[i])]
            reward, done, truncated, score = env.play_step(action_onehot)

            reward /= 10.0  # 보상 정규화: +10→+1, -10→-1, ±0.1→±0.01

            rewards[i] = reward
            dones[i] = done
            truncateds[i] = truncated

            if done:
                infos[i]['terminal_score'] = score
                if truncated:
                    # auto-reset 전에 terminal state 관측 보존 (GAE bootstrap용)
                    infos[i]['terminal_obs'] = env.get_grid_state()
                env.reset()

            states.append(env.get_grid_state())

        return np.stack(states, axis=0), rewards, dones, truncateds, infos

    def close(self):
        """인터페이스 통일용 no-op (단일 프로세스이므로 정리 불필요)."""
        pass

"""
멀티프로세스 벡터화 환경.

각 worker 프로세스가 1개의 SnakeGameAI를 소유하며
Pipe를 통해 명령을 주고받는다.

생성 순서 주의:
    SubprocVecEnv → PPOAgent (worker가 PyTorch 초기화 이전에 생성되어야 함)
"""
import multiprocessing as mp
import numpy as np

ACTION_MAP = [
    [1, 0, 0],  # 0: 직진
    [0, 1, 0],  # 1: 우회전
    [0, 0, 1],  # 2: 좌회전
]

BLOCK_SIZE = 20


def _worker(conn, grid_rows, grid_cols):
    """각 worker 프로세스의 진입점.

    명령어:
        ("step",   action_int)              → (state, reward, done, truncated, info)
        ("reset",  None)                    → state
        ("resize", (grid_rows, grid_cols))  → state (새 그리드 크기로 환경 재생성)
        ("close",  None)                    → 프로세스 종료
    """
    from game import SnakeGameAI
    env = SnakeGameAI(w=grid_cols * BLOCK_SIZE, h=grid_rows * BLOCK_SIZE, render=False)
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                action_onehot = ACTION_MAP[int(data)]
                reward, done, truncated, score = env.play_step(action_onehot)
                reward /= 10.0  # 보상 정규화: +10→+1, -10→-1

                info = {}
                if done:
                    info['terminal_score'] = score
                    if truncated:
                        info['terminal_obs'] = env.get_grid_state()
                    env.reset()

                state = env.get_grid_state()
                conn.send((state, reward, done, truncated, info))

            elif cmd == "reset":
                env.reset()
                conn.send(env.get_grid_state())

            elif cmd == "resize":
                new_rows, new_cols = data
                env = SnakeGameAI(
                    w=new_cols * BLOCK_SIZE,
                    h=new_rows * BLOCK_SIZE,
                    render=False,
                )
                conn.send(env.get_grid_state())

            elif cmd == "close":
                conn.close()
                break

    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        conn.close()


class SubprocVecEnv:
    """멀티프로세스 벡터화 환경.

    VecSnakeEnv와 동일한 reset()/step() API를 제공한다.
    render 인자는 무시되며 항상 headless로 동작한다.
    """

    def __init__(self, n_envs, render=False, grid_rows=24, grid_cols=32):
        if render:
            print("[SubprocVecEnv] 경고: 멀티프로세스 모드에서는 렌더링이 비활성화됩니다.")

        self.n_envs = n_envs
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self._closed = False

        self._parent_conns = []
        self._procs = []

        # macOS에서 fork는 deprecated이며, MPS 초기화 후 fork 시 간헐적 crash 가능
        # forkserver는 fork보다 안전하고 spawn보다 오버헤드 낮음
        ctx = mp.get_context("forkserver")
        for _ in range(n_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(child_conn, grid_rows, grid_cols),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # 부모에서는 child_conn 불필요
            self._parent_conns.append(parent_conn)
            self._procs.append(proc)

        # 결과 수집용 사전 할당 버퍼
        self._states = np.zeros((n_envs, 7, grid_rows, grid_cols), dtype=np.uint8)
        self._rewards = np.zeros(n_envs, dtype=np.float32)
        self._dones = np.zeros(n_envs, dtype=bool)
        self._truncateds = np.zeros(n_envs, dtype=bool)

    def reset(self):
        """모든 환경을 리셋하고 초기 상태를 반환한다.

        Returns:
            states: (n_envs, 7, grid_rows, grid_cols) uint8
        """
        for conn in self._parent_conns:
            conn.send(("reset", None))
        for i, conn in enumerate(self._parent_conns):
            self._states[i] = conn.recv()
        return self._states.copy()

    def step(self, actions):
        """모든 환경에서 action을 실행하고 결과를 반환한다.

        Args:
            actions: (n_envs,) int — 정수 인덱스 (0/1/2)

        Returns:
            states:     (n_envs, 7, grid_rows, grid_cols) uint8
            rewards:    (n_envs,) float32
            dones:      (n_envs,) bool
            truncateds: (n_envs,) bool
            infos:      list of dict
        """
        # 모든 worker에 동시 전송 (병렬 실행)
        for i, conn in enumerate(self._parent_conns):
            conn.send(("step", int(actions[i])))

        # 결과 수집
        infos = [{} for _ in range(self.n_envs)]
        for i, conn in enumerate(self._parent_conns):
            try:
                state, reward, done, truncated, info = conn.recv()
            except EOFError:
                raise RuntimeError(f"Worker {i} crashed unexpectedly.")
            self._states[i] = state
            self._rewards[i] = reward
            self._dones[i] = done
            self._truncateds[i] = truncated
            infos[i] = info

        return (
            self._states.copy(),
            self._rewards.copy(),
            self._dones.copy(),
            self._truncateds.copy(),
            infos,
        )

    def set_grid_size(self, grid_rows, grid_cols):
        """커리큘럼 승급 시 그리드 크기 변경.

        모든 worker에 resize 명령을 보내고 새 초기 상태를 수신한다.
        _states 버퍼도 새 크기로 재할당한다.

        Returns:
            states: (n_envs, 7, grid_rows, grid_cols) uint8 — 새 초기 상태
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        # 버퍼 재할당
        self._states = np.zeros((self.n_envs, 7, grid_rows, grid_cols), dtype=np.uint8)

        # 모든 worker에 resize 명령 전송
        for conn in self._parent_conns:
            conn.send(("resize", (grid_rows, grid_cols)))
        for i, conn in enumerate(self._parent_conns):
            self._states[i] = conn.recv()
        return self._states.copy()

    def close(self):
        """모든 worker 프로세스를 정리한다."""
        if self._closed:
            return
        self._closed = True
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=3)
            if proc.is_alive():
                proc.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

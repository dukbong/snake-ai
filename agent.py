import os
import numpy as np
import torch
from model import ActorCritic


class RolloutBuffer:
    """on-policy 경험을 저장하고 GAE를 계산하는 버퍼.

    고정 길이 rollout 후 GAE를 계산하고, 미니배치로 반환한다.
    상태는 메모리 절약을 위해 uint8로 저장하고, 모델 입력 시점에만 float32로 변환한다.
    """

    def __init__(self, n_steps, n_envs, obs_shape=(7, 24, 32)):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape

        # 사전 할당
        self.states = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=bool)
        self.truncateds = np.zeros((n_steps, n_envs), dtype=bool)
        # truncated 스텝의 V(s_terminal), 나머지는 0
        self.terminal_values = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.advantages = None
        self.returns = None

    def reallocate(self, obs_shape):
        """커리큘럼 승급 시 obs_shape 변경에 따른 버퍼 재할당."""
        self.obs_shape = obs_shape
        self.states = np.zeros((self.n_steps, self.n_envs, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
        self.log_probs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_envs), dtype=bool)
        self.truncateds = np.zeros((self.n_steps, self.n_envs), dtype=bool)
        self.terminal_values = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.advantages = None
        self.returns = None

    def store(self, step, states, actions, log_probs, rewards, values, dones, truncateds, terminal_values):
        """한 타임스텝 저장"""
        self.states[step] = states
        self.actions[step] = actions
        self.log_probs[step] = log_probs
        self.rewards[step] = rewards
        self.values[step] = values
        self.dones[step] = dones
        self.truncateds[step] = truncateds
        self.terminal_values[step] = terminal_values

    def compute_gae(self, last_values, last_dones, gamma, gae_lambda):
        """역방향 GAE 계산 → advantages, returns

        CleanRL 방식: dones[t]는 "스텝 t에서 done 발생"을 의미.
        truncated 스텝은 reward에 gamma * V(s_terminal)을 더해서 bootstrap 효과 반영.
        """
        advantages = np.zeros_like(self.rewards)
        last_gae = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
            else:
                next_values = self.values[t + 1]
                # dones[t]=True이면 스텝 t가 에피소드 마지막 → 새 에피소드 값 차단
                next_non_terminal = 1.0 - self.dones[t].astype(np.float32)

            # truncated 보정: done이지만 실제로 끝난 게 아닌 경우 V(s_terminal) bootstrap
            real_rewards = self.rewards[t].copy()
            real_rewards += gamma * self.terminal_values[t] * self.truncateds[t].astype(np.float32)

            delta = real_rewards + gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        self.returns = advantages + self.values

        # Advantage 정규화 (rollout 전체에 대해)
        adv_flat = advantages.reshape(-1)
        advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        self.advantages = advantages

    def get_batches(self, batch_size):
        """flatten → shuffle → 미니배치 generator"""
        total = self.n_steps * self.n_envs
        indices = np.random.permutation(total)

        states_flat = self.states.reshape(total, *self.obs_shape)
        actions_flat = self.actions.reshape(total)
        log_probs_flat = self.log_probs.reshape(total)
        advantages_flat = self.advantages.reshape(total)
        returns_flat = self.returns.reshape(total)

        for start in range(0, total, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield (
                states_flat[batch_idx],
                actions_flat[batch_idx],
                log_probs_flat[batch_idx],
                advantages_flat[batch_idx],
                returns_flat[batch_idx],
            )


class PPOAgent:
    """PPO 에이전트 (Actor-Critic)"""

    def __init__(
        self,
        n_envs=32,
        n_steps=256,
        n_epochs=3,
        batch_size=1024,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coeff=0.5,
        entropy_coeff=0.05,
        lr=3e-4,
        max_grad_norm=1.0,
        in_channels=7,
        obs_shape=(7, 24, 32),
        device=None,
    ):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.lr = lr
        self.max_grad_norm = max_grad_norm

        self.n_games = 0
        self.record = 0
        self.total_steps = 0

        # Device 설정: CUDA > MPS > CPU
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = ActorCritic(in_channels=in_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(n_steps, n_envs, obs_shape=obs_shape)

    def get_action(self, states):
        """rollout 수집용. no_grad로 실행."""
        states_tensor = torch.from_numpy(states).to(self.device)
        with torch.no_grad():
            actions, log_probs, _, values = self.model.get_action_and_value(states_tensor)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def update(self, last_values, last_dones, entropy_coeff=None, value_coeff=None):
        """GAE 계산 후 PPO 업데이트 수행.

        Args:
            last_values: (n_envs,) float32 — rollout 마지막 상태의 가치
            last_dones:  (n_envs,) bool   — rollout 마지막 스텝의 done 플래그
            entropy_coeff: float — 현재 entropy coefficient (None이면 self.entropy_coeff 사용)
            value_coeff: float — 현재 value coefficient (None이면 self.value_coeff 사용)

        Returns:
            metrics dict: policy_loss, value_loss, entropy, approx_kl
        """
        if entropy_coeff is None:
            entropy_coeff = self.entropy_coeff
        if value_coeff is None:
            value_coeff = self.value_coeff
        self.buffer.compute_gae(last_values, last_dones, self.gamma, self.gae_lambda)

        metrics = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'approx_kl': []}

        for _ in range(self.n_epochs):
            for states, actions, old_log_probs, advantages, returns in self.buffer.get_batches(self.batch_size):
                states_t = torch.from_numpy(states).to(self.device)
                actions_t = torch.from_numpy(actions).to(self.device)
                old_log_probs_t = torch.from_numpy(old_log_probs).to(self.device)
                advantages_t = torch.from_numpy(advantages).to(self.device)
                returns_t = torch.from_numpy(returns).to(self.device)

                _, new_log_probs, entropy, new_values = self.model.get_action_and_value(states_t, actions_t)

                ratio = torch.exp(new_log_probs - old_log_probs_t)

                # PPO clipped policy loss
                policy_loss = torch.max(
                    -ratio * advantages_t,
                    -torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_t
                ).mean()

                # Value loss
                value_loss = 0.5 * torch.nn.functional.mse_loss(new_values, returns_t)

                # Entropy bonus (음수: 최대화)
                entropy_loss = -entropy.mean()

                total_loss = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    # KL divergence 로깅 (early stopping 없음)
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(-entropy_loss.item())
                metrics['approx_kl'].append(approx_kl)

        return {k: np.mean(v) for k, v in metrics.items()}

    def save_checkpoint(self, file_name='model_ppo.pth', extra=None):
        model_folder_path = './model'
        os.makedirs(model_folder_path, exist_ok=True)
        path = os.path.join(model_folder_path, file_name)
        data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_games': self.n_games,
            'record': self.record,
            'total_steps': self.total_steps,
        }
        if extra:
            data.update(extra)
        torch.save(data, path)
        print(f'모델 저장 완료: {path} (게임 {self.n_games}회, 최고기록 {self.record}, 총 스텝 {self.total_steps})')

    def load_checkpoint(self, file_name='model_ppo.pth'):
        path = os.path.join('./model', file_name)
        if not os.path.exists(path):
            return None
        try:
            checkpoint = torch.load(path, weights_only=True, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.n_games = checkpoint.get('n_games', 0)
            self.record = checkpoint.get('record', 0)
            self.total_steps = checkpoint.get('total_steps', 0)
        except (RuntimeError, KeyError):
            print(f'체크포인트 아키텍처 불일치, 새로 학습 시작: {path}')
            return None
        print(f'이어서 학습: {path} (게임 {self.n_games}회, 최고기록 {self.record}, 총 스텝 {self.total_steps})')
        return checkpoint

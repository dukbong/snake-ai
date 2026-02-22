import torch
import torch.nn as nn


def _orthogonal_init(module, gain=1.0):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # 공유 backbone (7채널 입력)
        # 24×32 → 12×16
        self.conv1 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        # 12×16 → 6×8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        # 64 * 6 * 8 = 3072
        self.fc_shared = nn.Linear(3072, 256)

        # Policy head (Actor) — 3개 action에 대한 logits
        self.fc_policy = nn.Linear(256, 3)

        # Value head (Critic) — 상태 가치 스칼라
        self.fc_value = nn.Linear(256, 1)

        # Orthogonal 초기화
        _orthogonal_init(self.conv1, gain=(2 ** 0.5))
        _orthogonal_init(self.conv2, gain=(2 ** 0.5))
        _orthogonal_init(self.fc_shared, gain=(2 ** 0.5))
        _orthogonal_init(self.fc_policy, gain=0.01)
        _orthogonal_init(self.fc_value, gain=1.0)

    def _backbone(self, x):
        # uint8 → float32 변환 (호출자가 dtype을 신경 쓸 필요 없음)
        if x.dtype != torch.float32:
            x = x.float()
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc_shared(x))
        return x

    def forward(self, x):
        features = self._backbone(x)
        action_logits = self.fc_policy(features)
        value = self.fc_value(features)
        return action_logits, value

    def get_action_and_value(self, x, action=None):
        """rollout 수집 및 PPO 업데이트에서 공통 사용"""
        features = self._backbone(x)
        action_logits = self.fc_policy(features)
        value = self.fc_value(features).squeeze(-1)

        dist = torch.distributions.Categorical(logits=action_logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(self, x):
        """bootstrap value 계산 및 truncated terminal value 계산용"""
        features = self._backbone(x)
        return self.fc_value(features).squeeze(-1)

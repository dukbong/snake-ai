import torch
import torch.nn as nn
import torch.nn.functional as F


def _orthogonal_init(module, gain=1.0):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ResBlock(nn.Module):
    """Residual block + GroupNorm (RL에서 BN 불안정 방지)"""

    def __init__(self, channels):
        super().__init__()
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        _orthogonal_init(self.conv1, gain=(2 ** 0.5))
        _orthogonal_init(self.conv2, gain=(2 ** 0.5))

    def forward(self, x):
        residual = x
        x = F.relu(self.gn1(x))
        x = self.conv1(x)
        x = F.relu(self.gn2(x))
        x = self.conv2(x)
        return x + residual


class ActorCritic(nn.Module):
    def __init__(self, in_channels=7):
        super().__init__()
        # 입력 conv: in_channels → 128
        self.conv_in = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.gn_in = nn.GroupNorm(8, 128)

        # 3 Residual Blocks
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        self.res3 = ResBlock(128)

        # Downsample: stride-2 conv로 공간 차원을 절반으로 축소
        # 큰 맵에서 AdaptiveAvgPool의 급격한 압축 방지
        self.downsample = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
        )

        # AdaptiveAvgPool2d: 다운샘플된 feature map → 고정 (4, 4)
        # MPS에서 입력이 출력 크기(4)로 나누어떨어져야 하므로 forward에서 패딩 적용
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Shared FC: 128 * 4 * 4 = 2048
        self.fc_shared = nn.Linear(2048, 512)

        # Policy head (Actor) — 3개 action에 대한 logits
        self.fc_policy = nn.Linear(512, 3)

        # Value head (Critic) — 상태 가치 스칼라
        self.fc_value = nn.Linear(512, 1)

        # Orthogonal 초기화
        _orthogonal_init(self.conv_in, gain=(2 ** 0.5))
        _orthogonal_init(self.downsample[2], gain=(2 ** 0.5))
        _orthogonal_init(self.fc_shared, gain=(2 ** 0.5))
        _orthogonal_init(self.fc_policy, gain=0.01)
        _orthogonal_init(self.fc_value, gain=1.0)

    def _backbone(self, x):
        # uint8 → float32 변환 (호출자가 dtype을 신경 쓸 필요 없음)
        if x.dtype != torch.float32:
            x = x.float()
        x = x / 255.0
        x = F.relu(self.gn_in(self.conv_in(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.downsample(x)
        x = self.res3(x)
        # MPS 호환: AdaptiveAvgPool2d 입력이 출력 크기(4)로 나누어떨어져야 함
        h, w = x.shape[2], x.shape[3]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
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

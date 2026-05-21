"""Dueling DQN (Wang et al., ICML 2016) 정책/네트워크.

표준 DQN의 마지막 fully-connected 단을 두 갈래로 분리:

    V(s)    : 상태 가치(1차원)
    A(s,a)  : 행동 어드밴티지(n_actions 차원)

Q(s,a) = V(s) + ( A(s,a) - mean_a A(s,a) )

평균 빼주기는 V/A의 정체성 모호성(identifiability)을 해소한다.
"""

from __future__ import annotations

from typing import Any

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.dqn.policies import CnnPolicy, QNetwork


class DuelingQNetwork(QNetwork):
    """Q = V + (A - mean_a A) 의 dueling head."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        # QNetwork.__init__ 가 마지막에 self.q_net 을 만들지만, 우리는 이를 V/A 분기로 교체.
        # 부모 init 을 그대로 호출한 뒤 q_net 을 폐기 + 새 head 추가.
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch if net_arch else [512],
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )

        # Wang 2016 의 권고: V/A 모두 hidden 512 후 head 로 매핑.
        hidden = self.net_arch
        action_dim = int(self.action_space.n)
        # 기존 q_net 은 의미 없으니 비활성화 (저장/로드 호환 위해 빈 모듈 유지).
        self.q_net = nn.Identity()
        self.value_net = nn.Sequential(*create_mlp(self.features_dim, 1, hidden, self.activation_fn))
        self.advantage_net = nn.Sequential(
            *create_mlp(self.features_dim, action_dim, hidden, self.activation_fn)
        )

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        value = self.value_net(features)
        advantage = self.advantage_net(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingCnnPolicy(CnnPolicy):
    """DQNPolicy 의 ``make_q_net`` 만 DuelingQNetwork 로 교체."""

    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)


# SB3 가 정책을 string 으로 lookup 할 수 있도록 alias 도 제공
DuelingPolicy = DuelingCnnPolicy


def patch_default_features_extractor(policy_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Dueling 정책의 NatureCNN 기본값을 보장."""
    pk = dict(policy_kwargs or {})
    pk.setdefault("features_extractor_class", NatureCNN)
    return pk


# Schedule 은 ``BasePolicy._dummy_schedule`` 사용을 위해 export
_ = (BasePolicy, Schedule)

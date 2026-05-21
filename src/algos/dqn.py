"""DQN / Double DQN / Dueling DQN 빌더.

설정 예 (configs/*.yaml):

    algo:
      name: dqn
      features:
        double_q: true        # Double DQN target
        dueling: true         # Dueling Q head
      kwargs:
        policy: CnnPolicy
        learning_rate: 1.0e-4
        ...

세 가지 변형이 같은 코드 경로를 공유하도록 한다.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv

from src.algos.dueling import DuelingCnnPolicy


class DoubleDQN(DQN):
    """Double Q-learning target (van Hasselt et al., 2016).

    표준 DQN target:    y = r + γ * max_a' Q_target(s', a')
    Double DQN target:  y = r + γ * Q_target(s', argmax_a' Q_online(s', a'))

    SB3 ``DQN.train`` 을 거의 그대로 복제하고 next-Q 계산만 교체한다.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:  # noqa: D401
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts if replay_data.discounts is not None else self.gamma
            )

            with th.no_grad():
                # 온라인 네트워크가 next action 을 선택
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                # 타깃 네트워크가 그 액션의 Q-value 만 평가
                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = next_q_target.gather(1, next_actions)

                target_q_values = (
                    replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)) if losses else 0.0)


def _coerce_train_freq(value: Any) -> Any:
    """YAML 의 ``[4, step]`` 리스트를 SB3 의 ``(4, "step")`` 튜플로 보정."""
    if isinstance(value, list) and len(value) == 2:
        return (int(value[0]), str(value[1]))
    return value


def build_dqn(
    cfg: dict[str, Any],
    *,
    env: VecEnv,
    tensorboard_log: str | None,
    seed: int,
    device: str = "auto",
) -> DQN:
    """DQN/DDQN/Dueling DQN 인스턴스를 생성한다."""
    algo_cfg = cfg["algo"]
    features = dict(algo_cfg.get("features", {}) or {})
    kwargs = copy.deepcopy(algo_cfg.get("kwargs", {}) or {})

    use_double = bool(features.get("double_q", False))
    use_dueling = bool(features.get("dueling", False))

    # train_freq 튜플 보정
    if "train_freq" in kwargs:
        kwargs["train_freq"] = _coerce_train_freq(kwargs["train_freq"])

    # 정책 선택: dueling 이면 커스텀 정책 클래스로 교체
    if use_dueling:
        kwargs.pop("policy", None)
        policy: Any = DuelingCnnPolicy
    else:
        policy = kwargs.pop("policy", "CnnPolicy")

    # ``policy_kwargs.net_arch`` 같은 항목은 그대로 통과
    DqnClass: type[DQN] = DoubleDQN if use_double else DQN

    model = DqnClass(
        policy=policy,
        env=env,
        tensorboard_log=tensorboard_log,
        seed=seed,
        device=device,
        verbose=1,
        **kwargs,
    )
    return model

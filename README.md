# Breakout v5 Reinforcement Learning Final Project

ALE/Breakout-v5 환경에서 여러 강화학습 알고리즘을 학습·평가하고, 성능 차이를 이론적으로 해석하기 위한 프로젝트입니다. 과제의 핵심은 최고 점수 자체보다 **다양한 실험, 공정한 비교, 결과에 대한 강화학습 이론 기반 분석**입니다.

## Project Constraints

과제 공식 baseline 환경은 다음 패키지 버전을 고정합니다.

```bash
gymnasium[atari]==1.3.0
ale-py==0.11.2
autorom[accept-rom-license]==0.6.1
```

`docs/readme.md` 기준으로 필수 제약은 위 패키지 버전과 `ALE/Breakout-v5` 환경입니다. `batch_size`에 대한 고정 제약은 없습니다. 따라서 `batch_size`는 알고리즘별 표준 설정에 맞춰 다르게 둘 수 있습니다. 예를 들어 DQN의 `batch_size=32`는 replay buffer에서 샘플링하는 transition 수이고, PPO의 `batch_size=256`은 rollout batch를 나누는 minibatch 크기라 의미가 다릅니다.

Stable-Baselines3 PyPI 2.8.0은 `gymnasium<1.3.0` 제약이 있으므로, 이 프로젝트는 `gymnasium==1.3.0`과 호환되는 SB3 master 버전을 GitHub에서 설치합니다.

## Repository Structure

```text
Breakout_Final_Project_Pack/
├── README.md
├── requirements.txt
├── verify_env.ipynb
├── verify_env.py
├── docs/
│   ├── readme.md
│   └── requirements.txt
├── configs/
│   ├── dqn_baseline.yaml
│   ├── ddqn.yaml
│   ├── dueling_dqn.yaml
│   ├── a2c.yaml
│   ├── ppo.yaml
│   └── ablations/
│       ├── framestack_1.yaml
│       ├── reward_clip_off.yaml
│       └── sticky_action.yaml
├── src/
│   ├── envs.py
│   ├── eval.py
│   ├── render.py
│   ├── algos/
│   │   ├── registry.py
│   │   ├── dqn.py
│   │   ├── dueling.py
│   │   ├── a2c.py
│   │   └── ppo.py
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       └── seeding.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── play.py
│   └── plot_results.py
├── experiments/
└── reports/
    ├── figures/
    └── tables/
```

## Environment Setup

```bash
conda create -n breakout python=3.11 -y
conda activate breakout
pip install -r requirements.txt
AutoROM --accept-license
```

환경 확인:

```bash
python verify_env.py
```

또는 `verify_env.ipynb`를 열어 첫 번째 설치 셀과 검증 셀을 순서대로 실행합니다.

## Algorithms

현재 실험 파이프라인은 다음 알고리즘을 지원합니다.

| Algorithm | Config | Type | Main Purpose |
|---|---|---|---|
| DQN | `configs/dqn_baseline.yaml` | value-based, off-policy | Atari pixel control의 기본 baseline |
| Double DQN | `configs/ddqn.yaml` | value-based, off-policy | DQN의 Q-value overestimation 완화 |
| Dueling DQN | `configs/dueling_dqn.yaml` | value-based, off-policy | 상태 가치 `V(s)`와 advantage `A(s,a)` 분리 |
| A2C | `configs/a2c.yaml` | actor-critic, on-policy | A3C의 synchronous 변형, 단순 정책 기반 비교군 |
| PPO | `configs/ppo.yaml` | actor-critic, on-policy | clipped objective를 통한 안정적 policy optimization |

## Training

단일 학습:

```bash
python scripts/train.py --config configs/dqn_baseline.yaml --seed 7
python scripts/train.py --config configs/ddqn.yaml --seed 7
python scripts/train.py --config configs/dueling_dqn.yaml --seed 7
python scripts/train.py --config configs/a2c.yaml --seed 7
python scripts/train.py --config configs/ppo.yaml --seed 7
```

여러 seed 반복:

```bash
for s in 7 77 777; do
  python scripts/train.py --config configs/dqn_baseline.yaml --seed $s
  python scripts/train.py --config configs/ddqn.yaml --seed $s
  python scripts/train.py --config configs/dueling_dqn.yaml --seed $s
  python scripts/train.py --config configs/a2c.yaml --seed $s
  python scripts/train.py --config configs/ppo.yaml --seed $s
done
```

학습 결과는 `experiments/<timestamp>_<name>_seed<seed>/` 아래에 저장됩니다.

```text
config.yaml          # 실행 당시 config snapshot
tensorboard/         # TensorBoard logs
eval/                # EvalCallback outputs
best_model/          # best_model.zip
checkpoints/         # periodic checkpoints
final_model.zip      # final checkpoint
```

`--total-timesteps`로 config의 학습 step 수를 임시 override할 수 있습니다.

```bash
python scripts/train.py --config configs/ppo.yaml --seed 7 --total-timesteps 200000
```

## Evaluation

학습된 모델을 평가합니다. `--run`을 주면 `best_model/best_model.zip`을 우선 사용하고, 없으면 `final_model.zip`을 사용합니다.

```bash
python scripts/evaluate.py --run experiments/<run_dir> --n-eval-episodes 100
```

평가 환경은 학습 환경과 다르게 다음 기준을 사용합니다.

| Setting | Train | Evaluation |
|---|---|---|
| `terminal_on_life_loss` | `true` | `false` |
| `clip_reward` | `true` | `false` |
| policy | exploration 포함 가능 | deterministic 기본값 |

즉, 평가 점수는 목숨을 모두 소진할 때까지 플레이한 실제 episode return에 가깝습니다.

주요 평가 지표:

| Metric | Meaning |
|---|---|
| `mean` | 평균 episode return |
| `std` | episode return 표준편차 |
| `median` | 중앙값 |
| `min`, `max` | 최저/최고 episode return |
| `95% CI` | bootstrap 기반 평균 보상의 95% 신뢰구간 |
| `lengths` | 각 episode의 step 수 |

## Visualization

학습된 agent를 화면으로 확인합니다.

```bash
python scripts/play.py --run experiments/<run_dir> --n-episodes 3
python scripts/play.py --run experiments/<run_dir> --n-episodes 3 --window-scale 4
```

`--window-scale`을 쓰면 OpenCV 창으로 확대 렌더링합니다.

## Plotting Results

여러 실험 결과를 모아 learning curve, 최종 평가 분포, 요약 CSV를 생성합니다.

```bash
python scripts/plot_results.py
```

출력:

```text
reports/figures/learning_curves.png
reports/figures/eval_distribution.png
reports/tables/runs_summary.csv
```

## Suggested Experiment Plan

1. 환경 확인: `python verify_env.py`
2. 핵심 알고리즘 비교: DQN, Double DQN, Dueling DQN, A2C, PPO
3. 각 알고리즘 seed `7`, `77`, `777` 반복
4. 각 run에 대해 `--n-eval-episodes 100` 평가
5. ablation: frame stack 제거, reward clipping 제거, sticky action 적용
6. `plot_results.py`로 그림과 표 생성
7. 보고서에서 다음 관점으로 해석:
   - value-based vs actor-critic
   - off-policy vs on-policy
   - replay buffer와 target network의 안정화 효과
   - PPO clipping과 A2C의 단순 actor-critic 차이
   - preprocessing 변화가 partial observability와 reward scale에 미치는 영향

## References

### Environment and Tooling

- Bellemare, M. G., Naddaf, Y., Veness, J., & Bowling, M. (2013). The Arcade Learning Environment: An Evaluation Platform for General Agents. *Journal of Artificial Intelligence Research*, 47, 253-279.
- Towers, M., Terry, J. K., Kwiatkowski, A., et al. (2024). Gymnasium: A Standard Interface for Reinforcement Learning Environments. Farama Foundation.
- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *Journal of Machine Learning Research*, 22(268), 1-8.

### DQN Family

- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI Conference on Artificial Intelligence*.
- Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *International Conference on Machine Learning (ICML)*.

### Actor-Critic and Policy Optimization

- Mnih, V., Badia, A. P., Mirza, M., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. *International Conference on Machine Learning (ICML)*.  
  A3C paper; A2C is commonly treated as the synchronous, batched variant.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.

### Additional Useful References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Hessel, M., Modayil, J., van Hasselt, H., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. *AAAI Conference on Artificial Intelligence*.
# -RL-Breakout

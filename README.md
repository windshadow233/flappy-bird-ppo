# Flappy Bird with Reinforcement Learning（PPO Algorithm）

通过强化学习 PPO 算法玩 Flappy Bird。

```bash
git clone https://github.com/windshadow/flappy-bird-RL.git
cd flappy-bird-RL
pip install -r requirements.txt
```

## 训练

```bash
python train_ppo.py
```

## 测试

```bash
python test_ppo.py -ckpt ./ppo/models/best_model.pth -d -s 123456
```

- `-ckpt`：指定模型路径
- `-d`：是否显示游戏界面
- `-s`：随机种子

## 自己玩耍

```bash
python human_play.py
```
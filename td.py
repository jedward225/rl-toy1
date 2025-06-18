import time
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

"""
For Q_learning_step, Sarsa_step:
  nS: int, 状态数
  nA: int, 动作数
  gamma: float, 折扣因子 [0,1)
  alpha: float, 学习率
  state, next_state: int, 当前/下一个状态
  action, next_action: int, 当前/下一个动作
  reward: int, 0 或 1
  terminal: bool, 是否终止
"""

def epsilon_greedy_policy(nS, nA, Q_function, eps=0.5):
    """返回 epsilon-贪心策略矩阵，shape: [nS, nA]"""
    policy = np.zeros((nS, nA))
    for s in range(nS):
        # 基础概率
        policy[s, :] = eps / nA
        # 贪婪动作
        best_a = np.argmax(Q_function[s, :])
        policy[s, best_a] += 1.0 - eps
    return policy

def sample_action(policy, state):
    """根据给定策略分布 sample 一个动作"""
    return np.random.choice(policy.shape[1], p=policy[state])

def Q_learning_step(Q_function, state, action, reward, next_state, next_action, terminal, alpha, gamma):
    """
    Q-learning 更新:
    Q(s,a) ← Q(s,a) + alpha*(r + gamma*max_a' Q(s',a') - Q(s,a))
    """
    # 拷贝一份
    next_Q = Q_function.copy()
    # 目标值
    target = reward
    if not terminal:
        target += gamma * np.max(Q_function[next_state, :])
    # 更新
    next_Q[state, action] += alpha * (target - Q_function[state, action])
    return next_Q

def Sarsa_step(Q_function, state, action, reward, next_state, next_action, terminal, alpha, gamma):
    """
    SARSA 更新:
    Q(s,a) ← Q(s,a) + alpha*(r + gamma*Q(s',a') - Q(s,a))
    """
    next_Q = Q_function.copy()
    target = reward
    if not terminal:
        target += gamma * Q_function[next_state, next_action]
    next_Q[state, action] += alpha * (target - Q_function[state, action])
    return next_Q

def learn(learning_step, episodes=5000, max_steps=100, alpha=0.8, gamma=0.9):
    """
    执行 Q-learning 或 SARSA 训练，返回Q表和贪婪策略
    """
    env = gym.make('FrozenLake-v1', render_mode='ansi', is_slippery=False)
    env = TimeLimit(env, max_episode_steps=max_steps)

    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))

    # GLIE eps 衰减
    eps_decay = 1.0 / episodes

    for ep in range(episodes):
        state, _ = env.reset()
        terminal = False

        eps = max(0.01, 1.0 - ep * eps_decay)
        while not terminal:
            # 1. epsilon-贪心选择
            policy = epsilon_greedy_policy(nS, nA, Q, eps)
            action = sample_action(policy, state)

            # 2. 与环境交互
            next_state, reward, terminated, truncated, _ = env.step(action)
            terminal = terminated or truncated
            next_action = sample_action(policy, next_state)

            # 3. 更新 Q 表
            Q = learning_step(Q, state, action, reward, next_state, next_action, terminal, alpha, gamma)

            state = next_state

    # 最终策略为贪婪策略
    final_policy = np.argmax(Q, axis=1)
    return Q, final_policy

def render_single(env, policy, max_steps=100):
    """渲染一次策略执行过程"""
    episode_reward = 0
    state, _ = env.reset()
    for _ in range(max_steps):
        env.render()
        time.sleep(0.25)
        action = policy[state]
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break
    env.render()
    print("Episode reward: %.2f" % episode_reward)

if __name__ == "__main__":
    # Q-learning 训练 & 演示
    Q_q, p_q = learn(Q_learning_step, episodes=5000, alpha=0.8, gamma=0.9)
    print("Q-learning 最终策略:", p_q)
    env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=False)
    render_single(env, p_q)

    # SARSA 训练 & 演示
    Q_s, p_s = learn(Sarsa_step, episodes=5000, alpha=0.8, gamma=0.9)
    print("SARSA 最终策略:", p_s)
    render_single(env, p_s)

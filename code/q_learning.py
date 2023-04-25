
import numpy as np

# 初始化Q函数为0
Q = np.zeros((num_states, num_actions))

# 进行训练
for i in range(num_episodes):
    state = env.reset()
    while True:
        # 选择动作
        action = choose_action(Q, state)
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        # 更新Q函数
        Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        state = next_state
        if done:
            break
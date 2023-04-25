import numpy as np

# 初始化Q函数为0
Q = np.zeros((num_states, num_actions))

# 进行训练
for i in range(num_episodes):
    state = env.reset()
    action = choose_action(Q, state)
    while True:
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        # 选择下一个动作
        next_action = choose_action(Q, next_state)
        # 更新Q函数
        Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
        state = next_state
        action = next_action
        if done:
            break
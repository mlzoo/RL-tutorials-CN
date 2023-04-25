import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym
from tqdm import tqdm

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_v = nn.Linear(hidden_dim, 1)
        self.fc3_a = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        v = self.fc3_v(x)
        a = self.fc3_a(x)
        q = v + (a - torch.mean(a, dim=1, keepdim=True))
        return q

class DuelingDQN:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, epsilon, buffer_size, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.dqn = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_dqn = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_dqn.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

    def act(self, state):
        # e-greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q = self.dqn(state)
                return q.argmax(dim=1).item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        # sample出一个batch
        transitions = random.sample(self.buffer, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)
        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.uint8).unsqueeze(1).to(self.device)


        '''
        计算当前状态对应的Q值。
        首先，用self.dqn(state_batch)计算出当前状态state_batch下所有动作对应的Q值，得到一个大小为(batch_size, action_dim)的张量。
        然后，用action_batch中的每个动作对应的索引，从Q值张量中取出对应的Q值，得到一个大小为(batch_size, 1)的张量。
        这个操作可以使用PyTorch中的gather函数实现，其参数1表示在第1维进行索引。
        最终，得到一个大小为(batch_size, 1)的张量，即当前状态下执行对应动作的Q值。
        '''
        q = self.dqn(state_batch).gather(1, action_batch) #计算当前状态的Q值

        with torch.no_grad():
            '''
            下一个状态的Q值
            首先，使用目标神经网络self.target_dqn计算下一个状态next_state_batch下所有动作对应的Q值，得到一个大小为(batch_size, action_dim)的张量。
            然后，使用max函数，将Q值张量沿着第1维度（即所有动作的维度）取最大值，得到一个大小为(batch_size, 1)的张量，即下一个状态对应的最大Q值。
            最后，使用keepdim=True参数来保留张量的维度信息，以便后续计算。
            之所以取[0]，是因为它返回的是最大值和最大索引
            注意，这里的目标神经网络是通过软更新得到的，即每隔一段时间将当前神经网络的参数复制给目标神经网络，以此来稳定训练过程。
            '''
            next_q = self.target_dqn(next_state_batch).max(dim=1, keepdim=True)[0] #
            target_q = reward_batch + self.gamma * next_q * (1 - done_batch) 
        loss = nn.functional.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes):
        for i_episode in tqdm(range(num_episodes)):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.buffer.append((state, action, reward, next_state, done))
                state = next_state
                self.update()
                if (i_episode + 1) % 100 == 0:
                    env.render()
            if (i_episode + 1) % 10 == 0:
                
                self.target_dqn.load_state_dict(self.dqn.state_dict())

# 定义环境和参数
env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 128
lr = 0.001
gamma = 0.99
epsilon = 0.1
buffer_size = 10000
batch_size = 64
num_episodes = 500

# 创建Dueling DQN对象并进行训练
ddqn = DuelingDQN(state_dim, action_dim, hidden_dim, lr, gamma, epsilon, buffer_size, batch_size)
ddqn.train(env, num_episodes)

# 使用训练好的模型玩CartPole游戏
state = env.reset()
done = False
while not done:
    action = ddqn.act(state)
    state, reward, done, _ = env.step(action)
    env.render()
env.close()

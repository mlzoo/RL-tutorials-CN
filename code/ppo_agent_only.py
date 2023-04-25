# 导入必要的库
import gym  # 引入OpenAI Gym
import torch  # 引入PyTorch
import torch.nn as nn  # 引入PyTorch的神经网络模块
import torch.optim as optim  # 引入PyTorch的优化模块
import numpy as np  # 引入NumPy
from collections import deque  # 引入双端队列
from torch.distributions import Categorical
from tqdm import tqdm

# 定义神经网络模型
class Policy(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),  # 全连接层，输入为obs_size，输出为64
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(64, 64),  # 全连接层，输入为64，输出为64
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(64, act_size),  # 全连接层，输入为64，输出为act_size
            nn.Softmax(dim=-1)  # Softmax激活函数，dim=-1表示对最后一个维度进行Softmax
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)
# 定义PPO算法
class PPO:
    def __init__(self, obs_size, act_size, lr, gamma, clip_ratio):
        self.policy = Policy(obs_size, act_size)  # 初始化神经网络模型
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)  # 定义优化器
        self.gamma = gamma  # 定义折扣因子
        self.clip_ratio = clip_ratio  # 定义PPO中的超参数

    def update(self, rollouts):
        obs, act, rew, logp_old, adv = rollouts

        # 计算价值函数
        returns = np.zeros_like(rew)  # 初始化returns
        for t in reversed(range(len(rew))):
            if t == len(rew) - 1:
                returns[t] = rew[t]
            else:
                returns[t] = rew[t] + self.gamma * returns[t+1]  # 计算returns
        values = self.policy(torch.tensor(obs).float()).detach().numpy()  # 得到状态的价值
        adv = returns - np.sum(values, axis=1)
        
        # 计算旧策略的动作概率和对数概率
        act = torch.tensor(act).long()  # 将动作转换为Tensor类型
        logp_old = torch.tensor(logp_old).float()  # 将对数概率转换为Tensor类型
        pi_old = self.policy(obs).gather(1, act.unsqueeze(-1)).squeeze(-1)  # 得到旧策略的动作概率
        ratio = torch.exp(torch.log(pi_old) - logp_old)  # 计算比率
        surr1 = ratio * torch.from_numpy(adv).float()  # 第一项损失
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * torch.from_numpy(adv).float()  # 第二项损失
        loss = -torch.min(surr1, surr2).mean()  # PPO损失函数




        # 更新神经网络参数
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数

# 定义训练函数
def train(env_name, epochs, steps_per_epoch, batch_size, lr, gamma, clip_ratio):
    env = gym.make(env_name)  # 创建OpenAI Gym环境
    obs_size = env.observation_space.shape[0]  # 获取状态的维度
    act_size = env.action_space.n  # 获取动作的个数
    ppo = PPO(obs_size, act_size, lr, gamma, clip_ratio)  # 初始化PPO算法
    ep_reward = deque(maxlen=10)  # 初始化双端队列
    print('Started!')
    for epoch in range(epochs):
        print(epoch)
        obs_buf, act_buf, rew_buf, logp_buf = [], [], [], []  # 初始化存储buffer
        for _ in tqdm(range(steps_per_epoch)):
            
            obs = env.reset()  # 重置环境
            ep_reward.append(0)  # 初始化episode奖励
            for t in range(batch_size):
                probs = ppo.policy(torch.tensor(obs).float())
                m = Categorical(probs)
                act = m.sample()
                logp = m.log_prob(act)
                obs_buf.append(obs)  # 存储状态
                act_buf.append(act.item())  # 存储动作
                rew_buf.append(0)  # 存储奖励
                logp_buf.append(logp.item())  # 存储对数概率
                obs, rew, done, _ = env.step(act.item())  # 执行动作
                ep_reward[-1] += rew  # 更新episode奖励
                rew_buf[-1] += rew  # 更新奖励buffer
                if done:  # 如果终止，退出循环
                    break
            ppo.update((obs_buf, act_buf, rew_buf, logp_buf, np.zeros_like(rew_buf)))  # 更新策略
        print("Epoch: {}, Avg Reward: {:.2f}".format(epoch, np.mean(ep_reward)))  # 打印平均奖励

# 训练模型
if __name__ == '__main__':
    print('Go!Go!Go!')
    train('CartPole-v0', epochs=50, steps_per_epoch=4000, batch_size=128, lr=0.002, gamma=0.99, clip_ratio=0.2)

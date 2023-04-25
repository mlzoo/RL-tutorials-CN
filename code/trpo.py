import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from torch.distributions import Normal
from tqdm import tqdm

class ActorCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ActorCritic, self).__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_size),
            nn.Tanh()
        )

        self.critic_net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.actor_net(x), self.critic_net(x)

class TRPO:
    def __init__(self, obs_size, act_size, lr, gamma, delta, kl_target):
        self.actor_critic = ActorCritic(obs_size, act_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.delta = delta
        self.kl_target = kl_target

    def update(self, rollouts):
        obs, act, rew, logp_old, val_old = rollouts[:5]

        returns = np.zeros_like(rew)
        for t in reversed(range(len(rew))):
            if t == len(rew) - 1:
                returns[t] = rew[t]
            else:
                returns[t] = rew[t] + self.gamma * returns[t+1]


        values = self.actor_critic(torch.tensor(obs).float())[1].detach().numpy()
        adv = returns - np.sum(values, axis=1)

        act = torch.tensor(act).float()
        logp_old = torch.tensor(logp_old).float()
        mean, std = self.actor_critic(obs)
        dist = Normal(mean, std.abs() + 1e-8)
        pi_old = torch.exp(dist.log_prob(act))
        ratio = pi_old / (torch.exp(logp_old) + 1e-8)
        surr1 = ratio * torch.from_numpy(adv).float()
        surr2 = torch.clamp(ratio, 1 - self.delta, 1 + self.delta) * torch.from_numpy(adv).float()
        actor_loss = -torch.min(surr1, surr2).mean()

        val_old = torch.tensor(val_old).float()
        val = self.actor_critic(torch.tensor(obs).float())[1]
        critic_loss = nn.MSELoss()(val.squeeze(), torch.tensor(returns).float())

        with torch.no_grad():
            mean_old, std_old = self.actor_critic(obs)
            dist_old = Normal(mean_old, std_old.abs() + 1e-8)
            kl_div = torch.distributions.kl.kl_divergence(dist_old, dist).mean() # 计算新旧策略之间的KL散度

        # 更新网络，actor_loss和critic_loss不变，但加入了kl散度
        loss = actor_loss + 0.5 * critic_loss + self.kl_target * kl_div
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train(env_name, epochs, steps_per_epoch, batch_size, lr, gamma, delta, kl_target):
    env = gym.make(env_name)
    obs_size = env.observation_space.shape[0] # 2
    act_size = env.action_space.shape[0] # 1(连续型)

    trpo = TRPO(obs_size, act_size, lr, gamma, delta, kl_target)
    ep_reward = deque(maxlen=10)
    print('Started!')
    for epoch in range(epochs):
        obs_buf, act_buf, rew_buf, logp_buf = [], [], [], []
        for _ in tqdm(range(steps_per_epoch)):
            obs = env.reset()
            ep_reward.append(0)
            for t in range(batch_size):
                mean, std = trpo.actor_critic(torch.tensor(obs).float()) # 动作分布，均值和方差
                dist = Normal(mean, std.abs() + 1e-8) # 构建正态分布，和之前的Categorical是一个逻辑
                act = dist.sample() # 抽样
                logp = dist.log_prob(act) # 取log

                # 状态、动作、奖励、logp
                obs_buf.append(obs)
                act_buf.append(act.detach().numpy())
                rew_buf.append(0)
                logp_buf.append(logp.detach().numpy())

                # 走一步
                obs, rew, done, _ = env.step(act.detach().numpy())

                # 更新reward
                ep_reward[-1] += rew
                rew_buf[-1] += rew


                if done:
                    break

            # 更新actor-critic
            trpo.update((obs_buf, act_buf, rew_buf, logp_buf, np.zeros_like(rew_buf)))
        print("Epoch: {}, Avg Reward: {:.2f}".format(epoch, np.mean(ep_reward)))

if __name__ == '__main__':
    train('MountainCarContinuous-v0', epochs=50, steps_per_epoch=4000, batch_size=128, lr=0.002, gamma=0.99, delta=0.2,kl_target=0.02)
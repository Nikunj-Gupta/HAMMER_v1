import numpy as np


class GuessingSumEnv:
    def __init__(self, num_agents=5, discrete = True, scale=10.0):
        self.num_agents = num_agents
        self.discrete = discrete
        self.sum = 0
        self.scale = scale
        self.sum_scale = self.num_agents * self.scale
        self.agents = ["Agent{}".format(i) for i in range(self.num_agents)] 

    def step(self, actions):
        actions = np.array(list(actions.values())).reshape(-1, 1)
        if actions.shape != (self.num_agents, 1):
            raise Exception('got input shape ', actions.shape, ' instead of ', (self.num_agents, 1))

        observations = None
        rewards = -np.abs(actions - self.sum) # [-Inf ; 0]

        # normalized_rewards = (np.maximum(rewards, -self.sum_scale) + self.sum_scale) / self.sum_scale # [0 ; 1]
        normalized_rewards = rewards

        done = {}
        info = None

        rewards = {}
        for i, agent in enumerate(self.agents):
            rewards[agent] = normalized_rewards[i][0]
            done[agent] = True

        return observations, rewards, done, info

    def reset(self):
        if self.discrete:
            observations = np.random.randint(low=0, high= self.scale, size=(self.num_agents, 1))
        else: 
            observations = np.clip(np.random.normal(size=(self.num_agents, 1)), -self.scale, self.scale)
        self.sum = np.sum(observations)
        obs = {}
        for i, agent in enumerate(self.agents):
            obs[agent] = np.array(observations[i])
        return obs

    def render(self, mode='human'):
        return

    def close(self):
        return

    def seed(self, seed=None):
        np.random.seed(seed)
        return


if __name__ == '__main__':
    env = GuessingSumEnv()
    env.seed(0)

    print('obs:', env.reset())
    actions = {}
    for agent in env.agents:
        actions[agent] = np.random.randint(0, int(env.scale * env.num_agents), size=1)
    print('actions:', actions)
    print('Step Returns:', env.step(actions))
import gym

class GymEnv(object):
    def __init__(self, config):
        self.env = gym.make('CartPole-v0')
        self.state_size = self.env.observation_space.shape[0] # 4 dimension
        self.action_size = self.env.action_space.n # 2 actions

    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info
        
    def reset(self):
        self.env.reset()



if __name__ == '__main__':
    config = {}
    env = GymEnv(config)

from DQN import DQN
import gym
import numpy as np


def run():
    env = gym.make('CartPole-v0')
    env._max_episode_steps = None

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQN([state_size], action_size)
    # agent.load("./save/cartpole-dqn.h5")

    for e in range(agent.episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        for t in range(500):
            # env.render()
            np.reshape(state, [1, state_size])
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else 0
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            agent.replay(agent.batch_size)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                    .format(e, agent.episodes, t, agent.epsilon))
                break


if __name__ == '__main__':
    run()

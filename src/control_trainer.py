from DQN import DQN
import gym
import numpy as np


def atDestination():
    # Determine if we are finished or crashed

def reward(state, action):
    # Create reward policy given a state and action and possibly other variables

def control_and_update(state, action):
    # Send ROS message to perform action
    # Observe state in x_amount of seconds later...
    # Make sure action is correlated with change of state
    done = atDestination(state)
    reward = reward(state, action)
    info = debugInfo
    return next_state, reward, done, info

def run():
    # state_size = shape of state input (i.e. section of current occupancy grid (costmap))
    # action_size = shape of action output (i.e. slow down, speed up, stop, turn)
    agent = DQN(state_size, action_size)
    # agent.load("./save/model.h5")
    for e in range(agent.episodes):
        # TODO: reset state
        state = np.reshape(state, [1, 210, 160, 3])
        score = 0
        done = False
        while not done:
            score += 1
            # env.render()
            action = agent.act(state)
            # TODO: update state (i.e. observe)
            next_state, reward, done, info = control_and_update(state, action)
            reward = reward if not done else -100
            next_state = np.reshape(next_state, [1, 210, 160, 3])
            agent.remember(state, action, reward, next_state, done)
            agent.replay(agent.batch_size)
            state = next_state
            print "training."
            print "training.."
            print "training..."
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                .format(e, agent.episodes, score, agent.epsilon))

    # Save model
    model = agent.get_model()
    model.save('drqn_model.h5')


if __name__ == '__main__':
    run()

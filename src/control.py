from DQN import DRQN
import constants
import rospy
import numpy as np


def control(data):
    # Construct state representation
    global state_
    # Calculate values
    state = np.array([pedx-x, pedy-y, orthogonalVelocitiesDifference, pedTheta-Theta, headTheta])
    action = agent.act(state)

    msg = Byte()
    # Manage logic of deciding what to publish
    if action == hard:
        msg.data = '0x00'
    elif action == medium:
        msg.data = '0x01'
    else:
        msg.data = '0x02'

    pub.publish(msg)

    r.sleep()

if __name__ == '__main__':
    state_size = (GRID_WIDTH, GRID_HEIGHT, ENCODING_DIMENSIONS)
    # Light Medium Hard
    action_size = 3
    agent = DRQN(state_size, action_size)
    agent.load('drqn_model.h5', training=False)

    # ROS stuff
    rospy.init_node('dqn_control_node')
    r = rospy.Rate(10)
    sub = rospy.Subscriber("/caev/people", PeopleWithHead, control)
    pub = rospy.Publisher('dqn_control', Byte, queue_size=10)
    rospy.spin()

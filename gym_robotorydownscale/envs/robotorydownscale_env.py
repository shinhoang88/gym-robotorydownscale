import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import math
import random

import pybullet as p
import pybullet_data
import numpy as np

MAX_ESPISODE_LEN = 10 * 1000

# p.connect(p.GUI)
# urdfRootPath = pybullet_data.getDataPath()
# # Base plane
# planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"),
#                       basePosition=[0, 0, -0.115])
# # Add gravity
# p.setGravity(0, 0, -9.81)
# # Initial pose
# rest_pose = [0.61, 1.08, -0.23, 1.2, 0.26, 0.89, -2.81]
# # Load the URDF robot description
# downscaleUid = p.loadURDF(os.path.join(urdfRootPath, "downscale/downscale.urdf"),
#                                [0, 0, 0],
#                                useFixedBase=True)

class RobotorydownscaleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # Initialize the environment
    def __init__(self):
        self.step_counter = 0
        # Connect to pyBullet
        p.connect(p.GUI)
        # Adjust the view angle of the environment
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])
        # Joint velocity limits
        self.max_theta_prime = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.min_theta_prime = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]

        # Joint physical constraints
        self.max_theta = [3.14, 1.92, 3.14, 2.62, 3.14, 1.92, 3.14]
        self.min_theta = [-3.14, -1.92, -3.14, -2.62, -3.14, -1.92, -3.14]

        # Joint torque limits
        self.max_torque = [100, 250, 150, 150, 100, 100, 100]
        self.min_torque = [-100, -250, -150, -150, -100, -100, -100]

        # Define the box boundary for the Observation space
        # Observation [theta1, ... , theta7, theta1_prime, ... , theta7_prime]
        self.low_state = np.array([self.min_theta,
                                   self.min_theta_prime],
                                  dtype=np.float32)
        self.high_state = np.array([self.max_theta,
                                    self.max_theta_prime],
                                   dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state,
                                            dtype=np.float32)

        # Define the box boundary for the Action space
        # Action [theta1_prime, ... , theta7_prime]
        self.low_action = np.array(self.min_theta_prime,
                                   dtype=np.float32)
        self.high_action = np.array(self.max_theta_prime,
                                    dtype=np.float32)
        self.action_space = spaces.Box(low=self.low_action,
                                       high=self.high_action,
                                       dtype=np.float32)

    # Reset the environment
    def reset(self):
        self.step_counter = 0
        # Reset the pyBullet environment
        p.resetSimulation()
        # Disable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # Define URDF robot description path
        urdfRootPath = pybullet_data.getDataPath()
        # Base plane
        planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"),
                              basePosition=[0, 0, -0.115])
        # Add gravity
        p.setGravity(0, 0, -9.81)
        # Initial pose
        rest_pose = [0.61, 1.08, -0.23, 1.2, 0.26, 0.89, -2.81]
        # Load the URDF robot description
        self.downscaleUid = p.loadURDF(os.path.join(urdfRootPath, "downscale/downscale.urdf"),
                                       [0, 0, 0],
                                       useFixedBase=True)
        downscaleEndEffectorIndex = 6
        numJoints = p.getNumJoints(self.downscaleUid)

        # Initialize the robot to the init pose
        for i in range(numJoints):
            p.resetJointState(self.downscaleUid, i, rest_pose[i])

        # Update the robot current state
        joint_states = p.getJointStates(self.downscaleUid,
                                        range(p.getNumJoints(self.downscaleUid)))
        joint_infos = [p.getJointInfo(self.downscaleUid, i) for i in range(p.getNumJoints(self.downscaleUid))]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]

        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]

        # Observation [theta1, ... , theta7, theta1_prime, ... , theta7_prime]
        self.observation = joint_positions + joint_velocities

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Return the observation
        return self.observation

    def step(self, action):
        # Use this for better rendering
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # Get info from URDF file
        downscaleEndEffectorIndex = 6
        numJoints = p.getNumJoints(self.downscaleUid)

        # Define the target pose
        target_position = [0.6, -0.4, 0.2]
        target_orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi])
        target_joint_positions = \
            p.calculateInverseKinematics(self.downscaleUid,
                                         endEffectorLinkIndex=downscaleEndEffectorIndex,
                                         targetPosition=target_position,
                                         targetOrientation=target_orientation,
                                         lowerLimits=self.min_theta,
                                         upperLimits=self.max_theta,
                                         solver=p.IK_DLS)

        # Velocity control
        # Action = [tt_dot1, tt_dot2, ..., tt_dot7]
        for i in range(numJoints):
            p.setJointMotorControl2(bodyUniqueId=self.downscaleUid,
                                    jointIndex=i,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=action[i],
                                    velocityGain=1)
        # Step simulation
        p.stepSimulation()

        # Get new state data including: joint positions, velocities, torques
        state_robot = p.getLinkState(self.downscaleUid,
                                     downscaleEndEffectorIndex)[0]
        joint_states = p.getJointStates(self.downscaleUid,
                                        range(numJoints))
        joint_info = [p.getJointInfo(self.downscaleUid, i) for i in range(numJoints)]
        joint_states = [j for j, i in zip(joint_states, joint_info) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]

        cartesian_pos_error_threshold = 0.005

        # Target Cartesian postion indicator
        # If EE reach to the target -> indicator is set 0, else set -1
        if np.linalg.norm(np.array(state_robot) - np.array(target_position)) < cartesian_pos_error_threshold:
            cartesian_goal_indicator = 0
            done = True
        else:
            cartesian_goal_indicator = -1
            done = False

        # Joint limit indicator
        # If current joint position is over joint limit -> set -1, else set 0
        joint_penalty = [0, 0, 0, 0, 0, 0, 0]
        sum_joint_penalty = 0
        for i in range(numJoints):
            if (joint_positions[i] < self.min_theta[i]) | (joint_positions[i] > self.max_theta[i]):
                joint_penalty[i] = 1
            else:
                joint_penalty[i] = 0
            sum_joint_penalty = sum_joint_penalty + joint_penalty[i]
        if sum_joint_penalty == 0:
            joint_limit_indicator = 0  # It is safe
        else:
            joint_limit_indicator = -1  # Over Joint limit case

        # Norm of joint positions
        joint_positions_norm = np.linalg.norm(np.array(target_joint_positions) - np.array(joint_positions))
        # Norm of joint torques
        joint_torques_norm = np.linalg.norm(np.array(joint_torques))

        # Evaluate the reward:
        # Weighting factor
        w1 = -60  # joint position error weighting factor
        w2 = 100  # cartesian position indicator weighting factor
        w3 = 200  # joint limit penalty
        w4 = -2  # joint torque norm weighting factor
        w5 = -500  # Overall weighting factor

        reward = w1 * joint_positions_norm + \
                 w2 * cartesian_goal_indicator + \
                 w3 * joint_limit_indicator + \
                 w4 * joint_torques_norm + \
                 w5

        # Increase the counter
        self.step_counter += 1

        # If the step counter is over Max espisode length -> stop this episode
        if self.step_counter > MAX_ESPISODE_LEN:
            done = True

        info = state_robot
        self.observation = joint_positions + joint_velocities

        return self.observation, reward, done, info

    # def render(self, mode='human'):

    def __getstate__(self):
        return self.observation

    def close(self):
        p.disconnect()

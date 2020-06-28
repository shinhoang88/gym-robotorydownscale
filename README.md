## I) OpenAI Gym Environment for SKKU Robotory 7-DOF manipulator downscale version
There are 2 method for installing our Deep RL OpenAI Gym environment
### 1) Install directly:
- Download gym-robotorydownscale
- Move to this folder and:
```python
pip3 install gym-robotorydownscale
```
### 2) Building from source:
```python
git clone https://github.com/shinhoang88/gym-robotorydownscale.git
cd gym-robotorydownscale
pip3 install -e .
```
## II) Download the 7-DOF manipulator URDF description:
### 1) Source:
```python
git clone https://github.com/shinhoang88/downscale.git
```
### 2) Copy it into anaconda3, pybullet_data workspace:
```python
sudo cp -r /home/username/downscale/ /home/username/anaconda3/lib/python3.7/site-packages/pybullet_data/
```
## III) Test the environment working:
```python
import gym
import gym_robotorydownscale
import pybullet as p
import pybullet_data

# Initialize the OpenAI Gym environment
env = gym.make('robotorydownscale-v0')
for i_episode in range(20):
    # Reset the environment
    observation = env.reset()
    for t in range(18000):
        # Stochastic action sample inside the action_space box
        action = env.action_space.sample()
        # Getting info from environment step simulation
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()

```
- Author        : Phi Tien Hoang
- E-mail        : phitien@skku.edu
- Organization  : Robotory-SKKU-S.Korea


## OpenAI Gym Environment for SKKU Robotory 7-DOF manipulator downscale version
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
## Download the 7-DOF manipulator URDF description:
### 1) Source:
```python
git clone https://github.com/shinhoang88/downscale.git
```
### 2) Copy it into anaconda3, pybullet_data workspace:
```python
sudo cp -r /home/username/downscale/ /home/username/anaconda3/lib/python3.7/site-packages/pybullet_data/
```
- Author        : Phi Tien Hoang
- E-mail        : phitien@skku.edu
- Organization  : Robotory-SKKU-S.Korea


import setuptools
from pathlib import Path


setuptools.setup(
    name='gym_robotorydownscale',
    author="Phi Tien Hoang",
    author_email="phitien@skku.edu",
    version='0.0.1',
    description="A OpenAI Gym Env for Robotory downscale manipulator",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/shinhoang88/gym-robotorydownscale",
    packages=setuptools.find_packages(include="gym_robotorydownscale*"),
    install_requires=['gym', 'pybullet', 'numpy']  # And any other dependencies foo needs
)


# POENet

Deep learning is very flexible but sometimes requires intractably large amount of training data.
A common approach to address this is to restrict the models (e.g., neural nets) properly.
For example, imagine that we have to estimate an end-effector (6-dim.) pose from a image of a given robot (i.e., online hand-eye calibration).
It is quite obvious that end-to-end approaches (input: image, output: position/orientation)
violate the underlying physical constraints (e.g., inconsistent distances between adjacent joints).

In this project working with Seungyeon Kim and Younghun Kim ([SNU](https://sites.google.com/robotics.snu.ac.kr/fcp/)), we are exploring a way to inject such physical prior into learning-based modeling.
Specifically, our model estimates the underlying kinematic structure (i.e., joint twists) assuming a serial chain (or articulated rigid body), and
consistently gives the output poses that always satisfy the kinematics (i.e., product-of-exponential (POE)).
As preliminary results, we have identified the tendon-driven AMBIDEX wrist that has four joints actuated by two motors,
and currently working on image-to-pose problem with Frank Panda.

## Environment

This project is built on the pytorch template from [victoresque](https://github.com/victoresque/pytorch-template).
Verified that it works on Ubuntu 18.04, Python 3.9, GeForce RTX 2080 Ti, PyTorch 1.8.1 (other versions may not work).

## Terminal commands

Install:
```
$ conda create -n poenet python=3.9
$ conda activate poenet
$ pip install -r requirements.txt
```

Training:
```
$ python train.py --config config.json
$ python train.py --config config_image.json
```
These examples estimate the end-effector poses from the motor angles and the robot images, respectively.

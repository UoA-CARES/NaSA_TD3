# NaSA_TD3
Official PyTorch Implementation of Image-Based Deep Reinforcement Learning with Intrinsically Motivated Stimuli: On the Execution of Complex Robotic Tasks


## General Overview
See our  [Paper-Blog](https://sites.google.com/aucklanduni.ac.nz/nasa-td3-pytorch/home) for details  for pseudocode with more details of the training process as well as details of hyperparameters, full source code and videos of each task.


## Prerequisites

|Library         | Version |
|----------------------|----|
| Our RL Support Libray |[link](https://github.com/UoA-CARES/cares_reinforcement_learning)|
| DeepMind Control Suite |[link](https://github.com/deepmind/dm_control) |


## Network Architecture
<p align="center">
  <img src="https://github.com/UoA-CARES/NaSA_TD3/blob/main/repo_images/AE_TD3_network_diagram.png">
</p>


## Instructions Training
To train the NaSA-TD3 algorithm on the deep mind control suite from image-based observations, please run:
```
python3 train_loop.py --env=ball_in_cup --task=catch --seed=1 --intrinsic=True
```
## Our Results


<p align="center">
  <img src="https://github.com/UoA-CARES/NaSA_TD3/blob/main/repo_images/results_simulations.png">
</p>



## Citation
If you use either the paper or code in your paper or project, please kindly star this repo and cite our work.

---
title: Inverse Reinforcement Learning
---
So far we’ve manually designed a reward function in order to define a task

But what if we want ot learn hte reward function from observing expert- and then use RL

Couold apply approximate optimality model from before, but now learn the reward


Why worry about learning rewards?
- Biological basis in imitation learning: humans copy intent in imitation learning, standard imitation learning copies actions

Formally

Forward RL
- Given
	- States
	- Actions
	- Sometimes transitions
	- Reward Function
- Learn
	- $\pi^*(a|s)$

Inverse RL
- Given
	- States
	- Actions
	- Sometimes transitions
	- Sample trajectories from $\pi^*(a|s)$
- Learn
	- $r_\psi(s,a)$

Lots of reward function parameterization options
- (classic) Linear reward funtinon: weighted combination of features
- Neural network
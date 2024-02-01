---
title: Inverse Reinforcement Learning
---
So far we’ve always assumed we’ve had a reward function or manually designed on, in order to define a task.

What if we instead applied reinforcement learning to actually learn the reward function itself, by obvserving an expert?

**Why learn rewards?**
- Biological basis: humans copy intent in imitation learning, standard imitation learning copies actions
- Reinforcement learning: rewards are not always clear (e.g. self-driving cars)

So, **what is inverse RL**?
- Inferring reward functions from demonstrations
- Forward RL
	- Given states, actions, (sometimes) transitions, reward function
	- Learn  $\pi^*(a|s)$
- Inverse RL
	- Given states, actions, (sometimes) transitions, sample trajectories from $\pi^*(a|s)$
	- Learn $r_\psi(s,a)$

How do we construct a reward fuction?

We have some reward function parameterization options
- Linear: weighted combination of features
	- $r_\psi(s,a) = \sum_i \psi_if_i(s,a) = \psi^Tf(s,a)$ 
- Neural Network
	- $r_\psi(s,a)$ with some parameters $\psi$

### **Classical Approach to Inverse RL**

- We’re going to try to find the linear reward function $r_\psi(s,a) = \sum_i \psi_if_i(s,a) = \psi^Tf(s,a)$ 
- There was a **key idea**: if we know that the features $f_i$ are important, how about we try to match their expectations?
	- Let $\pi^{r_\psi}$ be the optimal policy for $r_\psi$
	- Pick $\psi$ such that $E_{\pi^{r_\psi}}[f(s,a)] = E_{\pi^*}[f(s,a)]$
	- vis. if you saw the expert driver rarely saw red lights, didn’t overtake people, etc. → matching the expected value of those features would give you similar behavior
- However, this is pretty ambigious- there are many ways you could have differnt $\psi$ vectors with equal expected values.
- So, how to disambiguate? One way is to use the **maximum margin principle**
	- Prety similar to the max marginal principle for SVM
	- Goal is to choose $\psi$ s.t. you maximize the margin between observed expert policy $\pi^*$ and all other policies
		- $\underset{\psi,m}{\text{max}}$ s.t. $\psi^TE_{\pi^*}[f(s,a)] \geq \underset{\pi \in \prod}\psi^TE_\pi[f(s,a)]+m$
		- Basically, find me weight vector $\psi$ such that the expert’s policy is better than all other policies by the largest possible margin
	- Still some issues- what if space of policies is large and continuous? Likely many polices that are basically equivalent to the experts. So, maybe weighgt by similarity between other policies and expert policies.
	- You can use the SVM trick here!

# Graphical Model

From now on, we’ll consider a probabilistic grpahical model of decision making, which means we’re basing our goal on finding the $O$ optimality variable

We can expresss this as a function of reward parameterized by $r_\psi$
- $p(O_t|s_t,a_t) = \text{exp}(r_\psi(s_t,a_t))$
- Goal: find $\psi$

We know that the probability of a trajectory given optimality and $\psi$ is 
- proportional to $p(\tau)\ \text{exp}(\sum_tr_\psi(s_t,a_t))$ 

Remember, in Inverse RL, we are
- given sampled $\tau$ from $\pi^*$


## Learning the Reward Function

How to learn $\psi$ for our reward function?
- Maximum likelihood learning!
- Maximize $1/N \sum_i \text{log}\ p(\tau_i|O_{1:T}, \psi)$
	- Which is equivalent to finding the maximimum of (ignoring $p(\tau)$ since is independent of $\psi$) $1/N \sum_i r_\psi(\tau_i) - \text{log}\ Z$ 
- Now what does this mean?
- Essentially, it says to pick parameters $\psi$ for $r_\psi$ such that we maximize the average reward plus a log normalizer (the partition function)

The partition function $Z$ = $\int p(\tau) \text{exp}(r_\psi(\tau))d\tau$ 


TODO

## Approximations in High Dimensions



## IRL and GANs



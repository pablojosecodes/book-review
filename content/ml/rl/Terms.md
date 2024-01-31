---
title: Terms
---

Here are some relevant basc terms. I’d recommend familiarizing yourself with what each of these mean, as they’ll come up often in these notes.

**Formal Representations**
- Agent: interacts with the environment
- $s$ current state
- $s’$ next state
- $r$ reward
- $a$ action
- $R$ reward function.
	- $r = R(s,a,s’)$ 
- $\tau$ trajectory
	- Sequence of states and actions
	- $\tau = (s_1,a_1, r_1, s_2, a_2, r_2, \dots)$
- $(s,a,r,s’)$ or $(s,a,s’)$ transition
- $h$ horizon
	- Length of a trajectory
- $R(\tau)$ reward
	- $R(\tau) = \sum r_i$  or $R(\tau) = \sum \gamma^ir_{i+1}$ 
	- Cumulative reward along trajectory $\tau$
-  $\gamma$ discount factor
- $\pi$ policy
	- Determine actions to be done
- $\pi(s)$ deterministic policy
	-  $a = \pi(s)$
- $\pi(a|s)$ stochastic policy
	- $P(a|s) = \pi(a|s)$
- $V(s)$ state value function
	- Expected return given state s and policy $\pi$
	- $V(s) = E[R(\tau)|s_1=s]$ 
	- Alternative notation $V^\pi(s) = E_{\tau \sim \pi}[R(\tau)|s_1=s]$
- $Q(s,a)$ action value function
	- Expected return given state s, action a, and policy $\pi$ 
	- $Q(s,a) = E[R(\tau)|s_1=s,a_1=1]$ 
	- Alternative notation: $Q^\pi(s,a) = E_{\tau \sim \pi}[R(\tau)|s_1=s,a_1=1]$
- $E[R(\tau)]$ expected return
	-  $E[R(\tau)] = \sum P(\tau)R(\tau)$ 
- $V^*(s)$ optimal state value function
	- Expected return if you start in state s and always act according to **optimal policy**
	- $V^*(s) = max_\pi \ V^\pi(s) = max_\pi \ E_{\tau \sim \pi} [R(\tau)|s_1=s]$ 
- $Q^*(s)$ optimal action value fucntion
	- Expected return if you start in state s, take action $a$, and always act according to **optimal policy**
	- $Q^*(s,a) = max_\pi \ Q^\pi(s) = max_\pi \ E_{\tau \sim \pi} [R(\tau)|s_1=s, a_1=a]$ 

**Bellman equations for optimal value functions** 
- Stochastic
	- $V^*(s) = \underset{a}{\text{max}} \sum_{s'} P(s'|s,a)(R(s,a) + \gamma V^*(s'))$ 
	- $Q^*(s,a) =  \sum_{s'} P(s'|s,a)[R(s,a) + \gamma \underset{a}{\text{max}} Q^*(s'])$ 
- Deterministic
	- $V^*(s) = \underset{a}{\text{max}}(R(s,a) + \gamma\ V^*(s’))$ 
	- $Q^*(s,a) = R(s,a)  + \gamma\  \underset{a'}{\text{max}}\ Q^*(s’,a'))$ 

**Task Types**
- Episodic task: limited length
- Continuing task: unlimited length

**Environments**
- Deterministic- only one possible transition $(s,a,s’)$ for a given $(s,a)$
- Non-deterministic- multiple transitions $(s,a,s’)$ for a given $(s,a)$
- Stochastic: non-deterministic with known probabilities for tansitions

**Action types**
- Discrete (categorical)
- Continuous (gaussian)

**Sequential decision problem**
- Analyze a sequence of actions based on expected rewards

**Markdov decision Process**: formalizes sequential decision problems in stochastic environments
- Set of states, actions, transition probability functions, and reward functions

**Markov property**
- Property such that the evolution of a markov process depends only on the present state

**Model-Based and Model-Free Methods**:
- **Model-Based Methods**: Algorithms that involve learning or using a model of the environment.
- **Model-Free Methods**: Algorithms that learn directly from interactions with the environment without an explicit model.

**Policy Iteration**:
- An algorithm for finding the optimal policy by iteratively improving the policy and evaluating it.

**Value Iteration**:
- An algorithm that successively improves the value function estimation and derives the optimal policy from it.

**Temporal Difference (TD) Learning**:
- A class of model-free methods that learn by bootstrapping from the current estimate of the value function.

**SARSA (State-Action-Reward-State-Action)**:
- An on-policy TD learning algorithm.
- Learns Q-values based on the action taken by the current policy.

**Off-Policy Learning**:
- Learning a policy different from the policy used to generate the data.

**Replay Buffer**:
- A data structure used to store and replay past experiences in order to break the temporal correlations in sequential data.

**Curriculum Learning**:
- A training methodology in reinforcement learning where tasks are gradually increased in complexity to facilitate learning.

**Partial Observability**:
- Describes environments where the agent does not have access to the complete state, leading to the Partially Observable Markov Decision Process (POMDP) framework.

**Reward Shaping**:
- Modifying the reward function to make learning faster or easier in reinforcement learning.



**Bellman equations for optimal value functions** 
- Stochastic
	- $V^*(s) = \underset{a}{\text{max}} \sum_{s'} P(s'|s,a)(R(s,a) + \gamma V^*(s'))$ 
	- $Q^*(s,a) =  \sum_{s'} P(s'|s,a)[R(s,a) + \gamma \underset{a}{\text{max}} Q^*(s'])$ 
- Deterministic
	- $V^*(s) = \underset{a}{\text{max}}(R(s,a) + \gamma\ V^*(s’))$ 
	- $Q^*(s,a) = R(s,a)  + \gamma\  \underset{a'}{\text{max}}\ Q^*(s’,a'))$ 

**Value Function**

How do you determine the value of a given state?
$V(s)$
- Notation for expected return $E[R(\tau)] = \sum P(\tau)R(\tau)$  for a stochastic policy $\tau$
- Value Function $V(s)$ quantifies value of a state (expected return given state s and policy $\pi$) 
	- $V(s) = E[R(\tau)|s_1=s]$ 
- Alternative notation (specify policy)
	- $V^\pi(s) = E_{\tau \sim \pi}[R(\tau)|s_1=s]$

How do you determine the value of a given action?
$Q(s,a)$
- Gives expected return
	- Specifying state $s$, action $a$, and policy $\pi$
- $Q(s,a) = E[R(\tau)|s_1=s,a_1=1]$ 
- Alternative notation: $Q^\pi(s,a) = E_{\tau \sim \pi}[R(\tau)|s_1=s,a_1=1]$



**Optimal value function** $V^*(s)$: expected return if you start in state s and always act according to optimal policy
-  $V^*(s) = max_\pi \ V^\pi(s) = max_\pi \ E_{\tau \sim \pi} [R(\tau)|s_1=s]$ 
**Optimal action-value function** $Q^*(s)$: expected return if you start in state s, take action a, and then act accordin to optimal policy in the environment
- $Q^*(s,a) = max_\pi \ Q^\pi(s) = max_\pi \ E_{\tau \sim \pi} [R(\tau)|s_1=s, a_1=a]$ 


#### **Value iteration**
**Key idea: using Bellman equations in practice**

*Obtaining the optimal policy given value function*
Using Q function
- $\pi^*(s) = argmax_a Q^*(s,a)$ 
Using the V function
- $\pi^*(s) = argmax_a \sum P(s’|s,a)(R(s,a,s’)+\gamma V^*(s’))$ 

**Value iteration algorithm**: estimate optimal $\pi$ 
*Approach*
1. Obtain optimal value function $Q^*(s,a) = R(s,a)  + \gamma\  \underset{a'}{\text{max}}\ Q^*(s’,a'))$
	1. Using Bellman equation
2. Obtain optimal policy from the obtained optimal value function 

*Algorithm*
Until Q doesn’t change, increase n from 1 by 1
- for each tuple of the transition model
	- $Q_n(s,a)$ ← $R(s,a) + \gamma\ \underset{a'}{\text{max}} Q_{n-1}(s',a')$
	- Note: $Q_n(s,a)$ is value $Q(s,a)$ at iteration $n$
Return $\pi$ such that $\pi(s) = \underset{a’}{\text{argmax}}\ Q_n(s,a’)$

Essentially, fill in  a Q table for each n, then returning a table for $\pi$ by picking the action which had the highest reward across the tables, for a given state

Designed to satisfy Bellman equation after multiple iterations
- $Q^*(s,a) = R(s,a)  + \gamma\  \underset{a'}{\text{max}}\ Q^*(s’,a'))$


## **Categorical Policy**: classifier over discrete actions. 
- Build nn for categorical policy like you would for classifier:
	- Input is observation
	- Some number of layers
	- Final linear layer (to get logits) and softmax (logits → probabilities)
- **Sampling**: Given probabilities for each action, PyTorch has built-in tools for sampling
- **Log likelihood**: Denote last layer of probabilities $P_\theta(s)$. Vector with as many entries as there are actions. Treat actions as indices for vector
	- Log likelihood for action $a$ is $log\ \pi_\theta(a|s) = log[P_\theta(s)]_a$ 

## **Diagonal Gaussian Policies**: 
**Normal Gaussian distribution**:
- We know multivariate Gaussian distribution is described by
	- mean vector $\mu$ (means of individual variables)
	- covariance matrix $\sum$ (covariance between each pair of variables in distribution)
		- $\Sigma = \begin{pmatrix}\sigma_1^2 & \sigma_{12} \\ \sigma_{21} & \sigma_2^2   \end{pmatrix}$
		- $\sigma_1^2$ :  variance of the first variable (which is the square of its standard deviation)
		- $\sigma_2^2$ : variance of the second variable
		-  $\sigma_{12}$ (or $\sigma_{21}$, as they are equal in a covariance matrix) is the covariance between the first and second variables.

**Diagonal Gaussian distribution**: special case where covariance matrix only has values on diagonal. 
- Implies no correlation between different variables- variables are independent from each other
- Essentialy  $\Sigma = \begin{pmatrix}\sigma_1^2 & 0 \\ 0 & \sigma_2^2   \end{pmatrix}$ , can also represent with a vector

**Diagonal Gaussian policy**

This policy always has a neural network that maps from observations to mean actions $\mu_\theta(s)$ 

2 ways the covariance matrix is represented
1. Single vector of  $\text{log } \sigma$ (log standard deviations)
	* Not a function of state: standalone parameters
2. Neural network which maps from states to $\text{log } \sigma_\theta(s)$
	* May share layers with the mean network

*Note*: we output log stds, not stds directly
- Why? Log stds take on any values in $(-\infty, \infty)$

**Sampling**
Given mean action $\mu_\theta(s)$, std $\sigma_\theta(s)$ and vector $z$ of noise from a spherical Gaussian $(z \sim N(0,I))$
- Action sample: $a = \mu_\theta(s) + \sigma_\theta(s) \odot z$ 

**Log Likelihood**
Log likelihood of 
- $k$-dimensional action $a$ 
- for diagonal Gaussian with
	- mean $\mu=\mu_\theta(s)$
	- std $\sigma= \sigma_\theta(s)$
Is
- $\log \pi_{\theta}(a|s) = -\frac{1}{2}\left(\sum_{i=1}^k \left(\frac{(a_i - \mu_i)^2}{\sigma_i^2} + 2 \log \sigma_i \right) + k \log 2\pi \right).$


**Bellman Equations**
- Key idea: value of your starting point is the reward you expect to get from being there plus wherever you land next
- The equations
	- $V_\pi(s) = E_{a \sim \pi , \ s’ \sim P}[r(s,a) + \gamma V^\pi(s’)]$ 
	- $Q_\pi(s) = E_{\ s’ \sim P}[r(s,a) + \gamma\  E_{a’ \sim \pi}[Q^\pi(s’,a’)]]$ 
	- $V^*(s) = \text{max}_a\ E_{a \sim \pi , \ s’ \sim P}\ [r(s,a) + \gamma V^*(s’)]$ 
	- $Q^*(s) = E_{\ s’ \sim P}\ [r(s,a) + \gamma\ \text{max}_a\  E_{a’ \sim \pi}[Q^*(s’,a’)]]$ 

**Bellman backup**: right hand side of Bellman equation reward + next value

**Advantage function**: sometimes we on’t need to describe how good an action is in an absolute sense, but how much better it is than others on average.
- ie. relative adnatage of that action
- Equation $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ 
- Corresponding to a policy $\pi$ describes how much better it is to take a specific action a in state s, over randomly selecting action according to $\pi(\cdot | s)$ assumign you act according to $\pi$ forever



# Goal of Reinforcement Learning

Background info
- probability of a given trajectory $p_\theta(\tau) = p(s_1) \prod_t \pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)$
## Finite Horizon
**Core idea**: sample from the state action marginal distribution, not from a trajectory distribution
- Reduces variance, more efficient, better for long horizons

 $\theta^* = \underset{\theta}{\text{arg max}} E_{\tau \sim p_\theta(\tau)}[\sum_t r(s_t, a_t)]$  
BECOMES
 $\theta^* = \underset{\theta}{\text{arg max}} \sum_t E_{(s_t,a_t) \sim p_\theta(s_t,a_t)}[r(s_t,a_t)]$ 
- Sampling from the *state action marginal* $p_\theta(s_t,a_t)$

## Infinite Horizon
Stationary distribution

Goal is still $\theta^* = \underset{\theta}{\text{arg max}} \sum_t E_{(s_t,a_t) \sim p_\theta(s_t,a_t)}[r(s_t,a_t)]$ 

If $T = \infty$, it  matters whether or not $p(s_t,a_t)$ converges to a stationary distribution
- ie $\mu = \mathrm{T}\mu$
- ie. $(T - I)\mu = 0$

For infinite horizons, we can consider $\theta^* = \underset{\theta}{\text{arg max}}   E_{(s_t,a_t) \sim p_\theta(s_t,a_t)}[r(s_t,a_t)]$
- No sum as we just want what’s expected in the end
# Anatomy of RL algorithms
1. Generate samples
2. Fit model / estimate return
3. Improve policy
## Value Functions

We know that our goal is to optimize for the estimated reward over a timestep
- Represented by $E_{\tau \sim p_\theta(\tau)}[\sum_t r(s_t,a_t)]$

How do we expand $E_{\tau \sim p_\theta(\tau)}[\sum_t r(s_t,a_t)]$?
- $E_{\tau \sim p_\theta(\tau)}[\sum_t r(s_t,a_t)]$ = $E_{s_1 \sim p(s_1)}[E_{a_1 \sim \pi(a_1|s_1)}[r(s_t,a_1) + E_{s_2 \sim \pi(s_2|s_1,a_1)}[ \ldots]]  ]$ 
- Expectation of state sampled from current probabillities of the expectation of action given  sampled state’s expectation of reward + expectation of state sampled from etc. 
**Action-State value function** We can represent part of this with $Q^\pi$ 
- State-action value function $Q^\pi(s_t,a_t) = \sum_t E_{\pi_\theta} [r(s_{t'}, a_{t'})|s_t,a_t]$  
- total reward from taking action $a_t$ in state $s_t$

**Action Value function** Value function $V^\pi$, same but given a state
- $V^\pi(s_t) = \sum_t E_{\pi_\theta} [r(s_{t'}, a_{t'})|s_t]$
- Equivalent to $V^\pi(s_t) =  E_{a_t \sim \pi(a_t|s_t)} [Q^\pi(s_t,a_t)]$ 

**Core ideas**
- **Idea 1**: You can improve a policy $\pi$ if you have $Q^\pi(s,a)$
- **Idea 2**: Compute gradient to increase probability of good $a$ given value function

# Rl algos review of types

**Core objective**: $\theta^* = \underset{\theta}{\text{arg max}}\ E_{\tau \sim p_\theta(\tau)}[\sum_t r(s_t,a_t)]$ 

Policy gradient: directly differentiate objective
Value based: estimate V/Q-function of optimal policy (no explicit policy however)
Actor-critic: Estimate V/Q-function of current policy, use to improve policy
MBRL: Estimate transition model + can either
- use for planning
- use to improve a policy
- something else

Why so many?
- Tradeoffs
	- Sample efficiency
	- Stability + ease of use
- Assumptions
	- Stochastic / deterministic
	-  Continuous / discrete
	- Horizon
- Somethigns are easy/hard in settings 
	- Difficulty representing model? Policy?
